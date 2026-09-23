"""导出 RMVPE 的 E2E0 网络为 ONNX（mel -> salience/hidden）。

说明：
  - 只导出神经网络部分（输入 log-mel (1,T,128)，输出 (1,T,360)）。
    音频 -> log-mel 由推理脚本用 numpy 完成（`mel_basis.npy` 随包提供，免 librosa 依赖）；
    hidden -> f0（cents 解码 + 插值）同样在脚本侧用 numpy 复刻。
  - E2E0 的 U-Net 要求帧数为 32 的倍数，故调用方需先把 mel pad 到 32 的倍数。

用法:
  python tools/export_onnx_rmvpe.py --out checkpoints/onnx_bundle/rmvpe.onnx
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

from modules.F0Predictor.rmvpe.constants import (MEL_FMAX, MEL_FMIN, N_MELS,  # noqa: E402
                                                 SAMPLE_RATE, WINDOW_LENGTH)
from modules.F0Predictor.rmvpe.model import E2E0  # noqa: E402

CKPT = "pretrain/rmvpe.pt"
HOP = 160


class E2E0Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):          # (B, 128, T) -> (B, T, 360)
        return self.model(mel)


def load_model(device="cpu"):
    m = E2E0(4, 1, (2, 2))
    ckpt = torch.load(CKPT, map_location=device)
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval()


def save_mel_basis(out_dir):
    from librosa.filters import mel as librosa_mel

    basis = librosa_mel(sr=SAMPLE_RATE, n_fft=WINDOW_LENGTH, n_mels=N_MELS,
                        fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True)
    path = os.path.join(out_dir, "mel_basis.npy")
    np.save(path, basis.astype(np.float32))
    print(f"  saved {path}  shape={basis.shape}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Export RMVPE E2E0 to ONNX")
    ap.add_argument("--out", default="checkpoints/onnx_bundle/rmvpe.onnx")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)

    print("[1/4] load RMVPE E2E0 ...")
    m = load_model("cpu")
    wrapper = E2E0Wrapper(m).eval()

    print("[2/4] export mel_basis + onnx ...")
    save_mel_basis(os.path.dirname(args.out))
    mel = torch.randn(1, N_MELS, 320)
    with torch.no_grad():
        ref = wrapper(mel)
        torch.onnx.export(
            wrapper, (mel,), args.out,
            input_names=["mel"], output_names=["hidden"],
            dynamic_axes={"mel": {2: "n_frames"}, "hidden": {1: "n_frames"}},
            opset_version=16, do_constant_folding=False, dynamo=False,
        )
    print(f"  saved {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    print("[3/4] verify with onnxruntime (multi length) ...")
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for t in (32, 160, 320, 640, 1280):
        x = torch.randn(1, N_MELS, t)
        with torch.no_grad():
            r = wrapper(x).numpy()
        g = sess.run(None, {"mel": x.numpy()})[0]
        print(f"  frames={t:5d} torch={r.shape} onnx={g.shape} "
              f"maxdiff={np.abs(r - g).max():.3e}")

    print("[4/4] write mel params for the inference package")
    info = {"sample_rate": SAMPLE_RATE, "n_fft": WINDOW_LENGTH, "hop_length": HOP,
            "win_length": WINDOW_LENGTH, "n_mels": N_MELS, "fmin": MEL_FMIN,
            "fmax": MEL_FMAX, "htk": True, "clamp": 1e-5, "pad_multiple": 32}
    p = os.path.join(os.path.dirname(args.out), "rmvpe_mel.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"  saved {p}")


if __name__ == "__main__":
    main()
