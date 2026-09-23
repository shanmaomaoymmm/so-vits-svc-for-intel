"""生成自包含 ONNX 推理包 checkpoints/onnx_bundle/。

包含：模型 onnx、numpy 需要的常量（mel 滤波器组、重采样卷积核、扩散系数）与运行配置。
目标机只需 `pip install onnxruntime numpy soundfile`。

用法:
  python tools/make_onnx_bundle.py
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

BUNDLE = os.path.join("checkpoints", "onnx_bundle")
SOVITS_DIR = os.path.join("checkpoints", "huawu_main_100000")
DIFF_DIR = os.path.join("checkpoints", "huawu_diff_188000")

SR = 44100
HOP = 512
ENC_SR = 16000


def copytree_files():
    os.makedirs(os.path.join(BUNDLE, "diffusion"), exist_ok=True)
    # 标准版（7 输入，MoeVoiceStudio 兼容）作为备选保留；
    # bundle 的 SoVits.onnx 由 tools/export_onnx_sovits_hifi.py 生成（随机外置、音质更好）。
    src = os.path.join(SOVITS_DIR, "huawu_main_100000_SoVits.onnx")
    shutil.copyfile(src, os.path.join(BUNDLE, "SoVits_moevs.onnx"))
    print("  copied SoVits_moevs.onnx (MoeVS compatible)")
    for f in os.listdir(DIFF_DIR):
        if f.endswith(".onnx"):
            short = f.replace("huawu_diff_188000_", "")
            shutil.copyfile(os.path.join(DIFF_DIR, f), os.path.join(BUNDLE, "diffusion", short))
            print(f"  copied diffusion/{short}")


def make_nsf_mel_basis():
    from librosa.filters import mel as librosa_mel

    basis = librosa_mel(sr=SR, n_fft=2048, n_mels=128, fmin=40, fmax=16000)
    np.save(os.path.join(BUNDLE, "nsf_mel_basis.npy"), basis.astype(np.float32))
    print(f"  nsf_mel_basis.npy {basis.shape}")


def make_resample_kernels():
    import torchaudio

    info = {}
    for w in (6, 128):
        r = torchaudio.transforms.Resample(SR, ENC_SR, lowpass_filter_width=w)
        k = r.kernel.detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(BUNDLE, f"resample_{SR}_{ENC_SR}_w{w}.npy"), k)
        info[str(w)] = int(r.width)
        print(f"  resample w={w}: kernel {k.shape} width={r.width}")
    return info


def make_diffusion_coeff():
    from diffusion.diffusion import GaussianDiffusion  # noqa: F401  (仅用于引用)
    from diffusion.diffusion_onnx import beta_schedule

    timesteps = 1000
    betas = beta_schedule["linear"](timesteps, max_beta=0.02)
    alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
    np.savez(os.path.join(BUNDLE, "diffusion_coeff.npz"),
             alphas_cumprod=alphas_cumprod.astype(np.float64),
             sqrt_alphas_cumprod=np.sqrt(alphas_cumprod).astype(np.float64),
             sqrt_one_minus_alphas_cumprod=np.sqrt(1 - alphas_cumprod).astype(np.float64))
    print(f"  diffusion_coeff.npz  alphas_cumprod[99]={alphas_cumprod[99]:.6f}")


def write_config(widths):
    cfg = {
        "sample_rate": SR,
        "hop_size": HOP,
        "encoder_sample_rate": ENC_SR,
        "speaker": "huawu",
        "speaker_id": 0,
        "noice_scale": 0.4,
        "pad_seconds": 0.5,
        "slice_db": -40,
        "resample": {
            "orig": SR, "new": ENC_SR,
            "kernel_w6": f"resample_{SR}_{ENC_SR}_w6.npy", "width_w6": widths["6"],
            "kernel_w128": f"resample_{SR}_{ENC_SR}_w128.npy", "width_w128": widths["128"],
        },
        "rmvpe": {
            "onnx": "rmvpe.onnx", "mel_basis": "mel_basis.npy", "mel": "rmvpe_mel.json",
            "thred": 0.05, "f0_min": 50, "f0_max": 1100,
        },
        "nsf_mel": {
            "mel_basis": "nsf_mel_basis.npy", "n_fft": 2048, "win_size": 2048,
            "hop_size": 512, "n_mels": 128, "fmin": 40, "fmax": 16000, "clip_val": 1e-5,
        },
        "sovits": {"onnx": "SoVits.onnx", "ssl_dim": 768, "uv_noise_dim": 192,
                   "vol_embedding": True, "use_f0_embed": True,
                   "dec_rand_dim": 17, "dec_noise_dim": 17, "upp": 512},
        "diffusion": {
            "dir": "diffusion", "k_step": 100, "speedup": 10, "t_start": 99,
            "spec_min": -12.0, "spec_max": 2.0, "coeff": "diffusion_coeff.npz",
            "sampler": "pndm",
        },
        "vocoder": {"onnx": "nsf_hifigan.onnx", "sine_dim": 9, "sine_amp": 0.1,
                    "noise_std": 0.003},
    }
    with open(os.path.join(BUNDLE, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("  config.json written")


def main():
    os.makedirs(BUNDLE, exist_ok=True)
    print("[1/5] copy onnx ...")
    copytree_files()
    print("[2/5] nsf mel basis ...")
    make_nsf_mel_basis()
    print("[3/5] resample kernels ...")
    widths = make_resample_kernels()
    print("[4/5] diffusion coeff ...")
    make_diffusion_coeff()
    print("[5/5] config.json ...")
    write_config(widths)

    print("\nbundle files:")
    for root, _, files in os.walk(BUNDLE):
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f"  {os.path.relpath(p, BUNDLE):40s} {os.path.getsize(p)/1e6:8.2f} MB")


if __name__ == "__main__":
    main()
