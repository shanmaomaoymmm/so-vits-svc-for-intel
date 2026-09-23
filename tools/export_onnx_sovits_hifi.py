"""导出"随机外置"版 SoVits ONNX（供自包含推理包使用）。

标准 `onnx_export.py` 产出的图里，dec(NSF 激励) 的随机量会被 torch.onnx trace 固化成常量，
导致 ONNX 输出带有固定噪声纹理（听感偏"沙"、频谱平坦度升高）。这里把这两个随机量外置为输入：

    c(1,T,768), f0(1,T), mel2ph(1,T), uv(1,T), noise(1,192,T), sid(1,), vol(1,T),
    rand_ini(1,H+1), dec_noise(1,T*512,H+1)  ->  audio(1,1,L)

调用方按原分布生成 rand_ini / dec_noise（第 1 列会被内部置 0）。

用法:
  python tools/export_onnx_sovits_hifi.py \
      --pth checkpoints/huawu_main_100000/model.pth \
      --config checkpoints/huawu_main_100000/config.json \
      --out checkpoints/onnx_bundle/SoVits.onnx
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

import utils  # noqa: E402
from models import f0_to_coarse  # noqa: E402
from onnxexport.model_onnx_speaker_mix import SynthesizerTrn  # noqa: E402


class SoVitsHifiWrapper(nn.Module):
    """复刻 onnxexport.SynthesizerTrn.forward，但把 dec 的随机量外置。"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, c, f0, mel2ph, uv, noise, sid, vol, rand_ini, dec_noise):
        net = self.net
        decoder_inp = F.pad(c, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, c.shape[-1]])
        c = torch.gather(decoder_inp, 1, mel2ph_).transpose(1, 2)

        if sid.dim() == 1:
            sid = sid.unsqueeze(0)
        g = net.emb_g(sid).transpose(1, 2)

        x_mask = torch.unsqueeze(torch.ones_like(f0), 1).to(c.dtype)
        vol_ = net.emb_vol(vol[:, :, None]).transpose(1, 2)
        x = net.pre(c) * x_mask + net.emb_uv(uv.long()).transpose(1, 2) + vol_

        z_p, m_p, logs_p, c_mask = net.enc_p(x, x_mask, f0=f0_to_coarse(f0), z=noise)
        z = net.flow(z_p, c_mask, g=g, reverse=True)
        o = net.dec(z * c_mask, f0, g=g, rand_ini=rand_ini, noise=dec_noise)
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default="checkpoints/huawu_main_100000/model.pth")
    ap.add_argument("--config", default="checkpoints/huawu_main_100000/config.json")
    ap.add_argument("--out", default="checkpoints/onnx_bundle/SoVits.onnx")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)

    print("[1/4] build SynthesizerTrn ...")
    hps = utils.get_hparams_from_file(args.config)
    net = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    utils.load_checkpoint(args.pth, net, None)
    net = net.eval()
    # 注意：这里刻意 **不** 调用 net.dec.OnnxExport()。
    # 该调用会让 SineGen 走专门为导出写的 "onnx" 分支，其中对超长序列做未取模的 cumsum，
    # fp32 下会产生明显相位误差（表现为直流偏置与频谱畸变）；而 PyTorch 生产推理走的是
    # `_f02sine` 分支（逐步取模，数值稳定）。保持 onnx=False 可让导出图与生产路径完全一致。
    for p in net.parameters():
        p.requires_grad = False
    wrapper = SoVitsHifiWrapper(net).eval()

    H = getattr(net.dec.m_source, "l_linear").in_features      # harmonic_num + 1
    T = args.frames
    L = T * int(np.prod(hps.model["upsample_rates"]))
    print(f"  harmonic dim={H}  T={T} -> samples={L}")

    c = torch.randn(1, T, hps.model["ssl_dim"])
    f0 = torch.rand(1, T) * 300 + 150
    f0[0, ::11] = 0
    mel2ph = (torch.arange(T) + 1).unsqueeze(0)
    uv = (f0 > 0).float()
    noise = torch.randn(1, hps.model["n_layers_trans_flow"] * 0 + 192, T)
    sid = torch.LongTensor([0])
    vol = torch.rand(1, T) * 0.1
    rand_ini = torch.rand(1, H)
    dec_noise = torch.randn(1, L, H)

    print("[2/4] self-check (torch, onnx branch) ...")
    with torch.no_grad():
        ref = wrapper(c, f0, mel2ph, uv, noise, sid, vol, rand_ini, dec_noise)
    print(f"  out {tuple(ref.shape)} rms={ref.pow(2).mean().sqrt().item():.4f}")

    print("[3/4] export onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (c, f0, mel2ph, uv, noise, sid, vol, rand_ini, dec_noise), args.out,
            input_names=["c", "f0", "mel2ph", "uv", "noise", "sid", "vol",
                         "rand_ini", "dec_noise"],
            output_names=["audio"],
            dynamic_axes={"c": {1: "n_frames"}, "f0": {1: "n_frames"},
                          "mel2ph": {1: "n_frames"}, "uv": {1: "n_frames"},
                          "noise": {2: "n_frames"}, "vol": {1: "n_frames"},
                          "dec_noise": {1: "n_samples"}, "audio": {2: "n_samples"}},
            opset_version=16, do_constant_folding=False, dynamo=False,
        )
    print(f"  saved {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    print("[4/4] verify with onnxruntime ...")
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feeds = {"c": c.numpy(), "f0": f0.numpy(), "mel2ph": mel2ph.numpy(), "uv": uv.numpy(),
             "noise": noise.numpy(), "sid": sid.numpy(), "vol": vol.numpy(),
             "rand_ini": rand_ini.numpy(), "dec_noise": dec_noise.numpy()}
    got = sess.run(None, feeds)[0]
    print(f"  ort {got.shape}  max_abs_diff={np.abs(got - ref.numpy()).max():.3e}  "
          f"cos={float(np.dot(got.ravel(), ref.numpy().ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref.numpy()))):.6f}")

    for t2 in (64, 137):
        c2 = torch.randn(1, t2, hps.model["ssl_dim"])
        f02 = torch.rand(1, t2) * 300 + 150
        m2 = (torch.arange(t2) + 1).unsqueeze(0)
        u2 = (f02 > 0).float()
        n2 = torch.randn(1, 192, t2)
        d2 = torch.randn(1, t2 * int(np.prod(hps.model["upsample_rates"])), H)
        with torch.no_grad():
            r2 = wrapper(c2, f02, m2, u2, n2, sid, vol[:, :t2], rand_ini, d2).numpy()
        g2 = sess.run(None, {"c": c2.numpy(), "f0": f02.numpy(), "mel2ph": m2.numpy(),
                             "uv": u2.numpy(), "noise": n2.numpy(), "sid": sid.numpy(),
                             "vol": vol[:, :t2].numpy(), "rand_ini": rand_ini.numpy(),
                             "dec_noise": d2.numpy()})[0]
        print(f"  frames={t2:4d} torch={r2.shape} onnx={g2.shape} "
              f"maxdiff={np.abs(r2 - g2).max():.3e}")


if __name__ == "__main__":
    main()
