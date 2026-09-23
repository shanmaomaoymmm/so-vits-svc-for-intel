"""导出 NSF-HiFiGAN 声码器为 ONNX（供浅层扩散路径使用）。

原 `SineGen.forward` 内部使用 `torch.rand` / `torch.randn_like`（随机初相与激励噪声），
直接 trace 会固化成常量或产生不支持的随机算子。这里把这两个随机量改为 **显式输入**：

    mel(1,128,T) + f0(1,T) + rand_ini(1,9) + noise(1,T*512,9) -> audio(1,1,T*512)

调用方按原分布生成 rand_ini / noise 即可，等价且可复现。

用法:
  python tools/export_onnx_vocoder.py --out checkpoints/onnx_bundle/nsf_hifigan.onnx
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

LRELU_SLOPE = 0.1
MODEL_PATH = "pretrain/nsf_hifigan/model"


class VocoderWrapper(nn.Module):
    """复刻 `Generator.forward`，但把 SineGen 的随机量外置为输入。"""

    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        sg = gen.m_source.l_sin_gen
        self.sine_amp = sg.sine_amp
        self.noise_std = sg.noise_std
        self.sr = sg.sampling_rate
        self.dim = sg.dim
        self.upp = gen.upp

    def forward(self, mel, f0, rand_ini, noise):
        g = self.gen
        f0_ = f0.unsqueeze(-1)                                  # (B, T, 1)
        harm = torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device)
        fn = f0_ * harm.reshape(1, 1, -1)
        rad = (fn / self.sr) % 1
        rad = rad + torch.cat([torch.zeros_like(rand_ini[:, :1]), rand_ini[:, 1:]], dim=1)
        tmp = torch.cumsum(rad, 1) * self.upp
        tmp = F.interpolate(tmp.transpose(2, 1), scale_factor=self.upp,
                            mode="linear", align_corners=True).transpose(2, 1)
        rad = F.interpolate(rad.transpose(2, 1), scale_factor=self.upp,
                            mode="nearest").transpose(2, 1)
        tmp = tmp % 1
        idx = (tmp[:, 1:, :] - tmp[:, :-1, :]) < 0
        shift = torch.zeros_like(rad)
        shift[:, 1:, :] = idx * -1.0
        sine = torch.sin(torch.cumsum(rad + shift, dim=1) * 2 * math.pi) * self.sine_amp
        uv = (f0_ > 0).to(f0.dtype)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=self.upp,
                           mode="nearest").transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        sine = sine * uv + noise_amp * noise
        har_source = g.m_source.l_tanh(g.m_source.l_linear(sine)).transpose(1, 2)  # (B,1,L)

        x = g.conv_pre(mel)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = g.ups[i](x)
            x = x + g.noise_convs[i](har_source)
            xs = None
            for j in range(g.num_kernels):
                rb = g.resblocks[i * g.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / g.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        return torch.tanh(g.conv_post(x))


def load_generator(device="cpu"):
    from vdecoder.nsf_hifigan.models import load_model

    gen, h = load_model(MODEL_PATH, device=device)
    return gen.eval(), h


def main():
    ap = argparse.ArgumentParser(description="Export NSF-HiFiGAN vocoder to ONNX")
    ap.add_argument("--out", default="checkpoints/onnx_bundle/nsf_hifigan.onnx")
    ap.add_argument("--frames", type=int, default=200, help="导出/自检用的 mel 帧数")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)

    print("[1/4] load generator ...")
    gen, h = load_generator("cpu")
    wrapper = VocoderWrapper(gen).eval()
    print(f"  num_mels={h.num_mels} sr={h.sampling_rate} hop={h.hop_size} dim={wrapper.dim}")

    T = args.frames
    L = T * wrapper.upp
    mel = torch.randn(1, h.num_mels, T) * 0.5
    f0 = torch.rand(1, T) * 400 + 100
    f0[0, ::7] = 0.0                       # 模拟清音
    rand_ini = torch.rand(1, wrapper.dim)
    noise = torch.randn(1, L, wrapper.dim)

    print("[2/4] self-check (torch wrapper) ...")
    with torch.no_grad():
        ref = wrapper(mel, f0, rand_ini, noise)
    print(f"  out {tuple(ref.shape)}  rms={ref.pow(2).mean().sqrt().item():.4f}")

    print("[3/4] export onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (mel, f0, rand_ini, noise), args.out,
            input_names=["mel", "f0", "rand_ini", "noise"], output_names=["audio"],
            dynamic_axes={"mel": {2: "n_frames"}, "f0": {1: "n_frames"},
                          "noise": {1: "n_samples"}, "audio": {2: "n_samples"}},
            opset_version=16, do_constant_folding=False, dynamo=False,
        )
    print(f"  saved {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    print("[4/4] verify with onnxruntime")
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"mel": mel.numpy(), "f0": f0.numpy(),
                          "rand_ini": rand_ini.numpy(), "noise": noise.numpy()})[0]
    print(f"  ort {got.shape}  max_abs_diff={np.abs(got - ref.numpy()).max():.3e}")

    # 不同帧数（动态维）验证
    for t2 in (50, 137, 400):
        m2 = torch.randn(1, h.num_mels, t2) * 0.5
        f2 = torch.rand(1, t2) * 400 + 100
        n2 = torch.randn(1, t2 * wrapper.upp, wrapper.dim)
        with torch.no_grad():
            r2 = wrapper(m2, f2, rand_ini, n2).numpy()
        g2 = sess.run(None, {"mel": m2.numpy(), "f0": f2.numpy(),
                             "rand_ini": rand_ini.numpy(), "noise": n2.numpy()})[0]
        print(f"  frames={t2:4d} torch={r2.shape} onnx={g2.shape} "
              f"maxdiff={np.abs(r2 - g2).max():.3e}")


if __name__ == "__main__":
    main()
