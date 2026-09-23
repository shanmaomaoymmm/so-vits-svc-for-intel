"""
自转换保真度评估（模型选型用）
==============================
以说话人自身的**留出集音频**为输入与参考，做同一说话人自转换（self-conversion），
衡量各候选模型在“尽量还原原音频”这一目标下的保真度：

  - mel_L1_dB  : 输出与参考的 80 维 log-mel 平均绝对误差（dB，越低越保真）
  - MCD        : 梅尔倒谱失真（越低越保真）
  - dHNR_dB    : 输出 HNR - 参考 HNR（>0 更“干净”，<0 更“气声/毛刺”）
  - dCentroid  : 输出谱质心 - 参考谱质心（Hz，>0 更亮）

用法:
  python tools/self_convert_eval.py \
      --case main_100000=logs/44k/G_100000.pth=logs/44k/config.json \
      --case main_98000=logs/44k/G_98000.pth=logs/44k/config.json --n 3
  # 浅层扩散对比
  python tools/self_convert_eval.py --shd --diffusion logs/44k/diffusion/model_150000.pt \
      --diffusion-config logs/44k/diffusion/config.yaml \
      --case g98_shd150=logs/44k/G_98000.pth=logs/44k/config.json
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.append(os.getcwd())
from inference.infer_tool import Svc  # noqa: E402

FRAME = 2048
HOP = 512
SR = 44100


def load_mono(path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float64), sr


def log_mel(x, n_mels=80, fmax=22050):
    import librosa

    m = librosa.feature.melspectrogram(
        y=x.astype(np.float32), sr=SR, n_fft=FRAME, hop_length=HOP,
        win_length=FRAME, n_mels=n_mels, fmin=0.0, fmax=fmax, power=2.0)
    return 10.0 * np.log10(np.maximum(m, 1e-10))


def mcd(mel_a, mel_b, n_dct=13):
    from scipy.fftpack import dct

    ca = dct(mel_a, type=2, axis=0, norm="ortho")[:n_dct]
    cb = dct(mel_b, type=2, axis=0, norm="ortho")[:n_dct]
    d = ca - cb
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(d ** 2, axis=0))))


def hnr_median(x):
    lag_min, lag_max = int(SR / 500), int(SR / 70)
    win = np.hanning(FRAME)
    vals = []
    for i in range(0, max(1, len(x) - FRAME), HOP):
        fr = x[i:i + FRAME]
        if len(fr) < FRAME:
            break
        if np.sqrt(np.mean(fr ** 2)) < 1e-4:
            continue
        fr = (fr - fr.mean()) * win
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        ac = ac / (ac[0] + 1e-12)
        seg = ac[lag_min:lag_max]
        if len(seg) == 0:
            continue
        r = min(max(float(seg.max()), 1e-4), 0.9999)
        vals.append(10 * np.log10(r / (1 - r)))
    return float(np.median(vals)) if vals else float("nan")


def centroid(x):
    win = np.hanning(FRAME)
    P, cnt = None, 0
    for i in range(0, max(1, len(x) - FRAME), HOP):
        fr = x[i:i + FRAME] * win
        if np.sqrt(np.mean(fr ** 2)) < 1e-4:
            continue
        S = np.abs(np.fft.rfft(fr)) ** 2
        P = S if P is None else P + S
        cnt += 1
    if P is None:
        return float("nan")
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    Pn = P / P.sum()
    return float((freqs * Pn).sum())


def evaluate(svc, ref_path, f0p, k_step, second_encoding=False):
    ref, _ = load_mono(ref_path)
    kwarg = {
        "raw_audio_path": ref_path, "spk": "huawu", "tran": 0,
        "slice_db": -40, "cluster_infer_ratio": 0, "auto_predict_f0": False,
        "noice_scale": 0.4, "pad_seconds": 0.5, "clip_seconds": 0,
        "lg_num": 0, "lgr_num": 0.75, "f0_predictor": f0p,
        "enhancer_adaptive_key": 0, "cr_threshold": 0.05, "k_step": k_step,
        "use_spk_mix": False, "second_encoding": second_encoding,
        "loudness_envelope_adjustment": 1,
    }
    out = svc.slice_inference(**kwarg)
    out = np.asarray(out, dtype=np.float64)
    if out.ndim > 1:
        out = out.mean(axis=1)
    n = min(len(out), len(ref))
    out, ref = out[:n], ref[:n]
    ma, mb = log_mel(out), log_mel(ref)
    w = min(ma.shape[1], mb.shape[1])
    ma, mb = ma[:, :w], mb[:, :w]
    return {
        "mel_L1_dB": float(np.mean(np.abs(ma - mb))),
        "MCD": mcd(ma, mb),
        "hnr_out": hnr_median(out),
        "hnr_ref": hnr_median(ref),
        "cent_out": centroid(out),
        "cent_ref": centroid(ref),
        "dur_s": round(n / SR, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Self-conversion fidelity evaluation")
    ap.add_argument("--case", action="append", required=True,
                    help="label=model=config[=diffusion_model=diffusion_config] (repeatable)")
    ap.add_argument("--refs", default=os.path.join("filelists", "val.txt"))
    ap.add_argument("--n", type=int, default=3, help="number of held-out refs")
    ap.add_argument("--f0p", default="rmvpe")
    ap.add_argument("--vocoder-device", default="cpu")
    ap.add_argument("--diffusion", default=None)
    ap.add_argument("--diffusion-config", default=None)
    ap.add_argument("--k-step", type=int, default=100)
    ap.add_argument("--shd", action="store_true")
    args = ap.parse_args()

    with open(args.refs, encoding="utf-8") as f:
        refs = [ln.strip() for ln in f if ln.strip()][:args.n]
    print(f"refs ({len(refs)}): {refs}")

    print(f"\n{'model':>16} | {'mel_L1_dB':>9} {'MCD':>7} {'dHNR':>7} "
          f"{'dCent':>8}  |  per-ref mel_L1")
    print("-" * 100)
    for spec in args.case:
        parts = spec.split("=")
        label, model, config = parts[0], parts[1], parts[2]
        diff = parts[3] if len(parts) > 3 and parts[3] else args.diffusion
        dconf = parts[4] if len(parts) > 4 and parts[4] else args.diffusion_config
        shd = args.shd or bool(diff)
        svc = Svc(model, config, None, "", False, diff,
                  dconf, shd, False, False, False,
                  vocoder_device=args.vocoder_device, mono_mode=True)
        rows = []
        for r in refs:
            rows.append(evaluate(svc, r, args.f0p, args.k_step))
            svc.clear_empty()
        mel = np.mean([x["mel_L1_dB"] for x in rows])
        mcdv = np.mean([x["MCD"] for x in rows])
        dhnr = np.nanmean([x["hnr_out"] - x["hnr_ref"] for x in rows])
        dcent = np.nanmean([x["cent_out"] - x["cent_ref"] for x in rows])
        per = " ".join(f"{x['mel_L1_dB']:.2f}" for x in rows)
        print(f"{label:>16} | {mel:>9.2f} {mcdv:>7.2f} {dhnr:>+7.2f} "
              f"{dcent:>+8.0f}  |  {per}")
        del svc


if __name__ == "__main__":
    main()
