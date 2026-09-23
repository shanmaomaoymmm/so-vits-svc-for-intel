"""
推理输出音频质量评估工具
========================
两种模式:

1) ab —— 多个音频的客观指标对比（响度/动态/削波/底噪/频谱质心/谱平坦度/HNR 等），
   并给出两两之间的整体差异与分频段能量差。适合比较不同 checkpoint / 不同参数的推理结果。

2) timbre —— 与训练集音色的对比（以训练集长时平均频谱 LTAS 为参考），
   输出 1/3 倍频程差值、谱倾斜与音色距离，用于判断输出相对目标音色是否更闷/更亮。

用法:
  python tools/compare_audio.py ab results/a.wav results/b.wav
  python tools/compare_audio.py ab results/a.wav results/b.wav results/c.wav
  python tools/compare_audio.py timbre results/a.wav results/b.wav --train-list filelists/train.txt --n 80

注: 输出文本使用英文以避免 Windows 终端编码导致的乱码。
"""
import argparse
import os
import random
import sys

import numpy as np
import soundfile as sf

FRAME = 2048
HOP = 512
TRAIN_LIST = os.path.join("filelists", "train.txt")

# 1/3 octave band centers
CENTERS = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
           1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]


# --------------------------------------------------------------------------
# common helpers
# --------------------------------------------------------------------------
def load(path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float64), sr


def avg_power_spectrum(x, sr, max_frames=None):
    """Long-term average power spectrum (silent frames excluded)."""
    win = np.hanning(FRAME)
    P, cnt = None, 0
    for i in range(0, max(0, len(x) - FRAME), HOP):
        fr = x[i:i + FRAME] * win
        if np.sqrt(np.mean(fr ** 2)) < 1e-4:
            continue
        S = np.abs(np.fft.rfft(fr)) ** 2
        P = S if P is None else P + S
        cnt += 1
        if max_frames and cnt >= max_frames:
            break
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
    if P is None:
        return np.zeros_like(freqs), freqs
    return P / cnt, freqs


def norm(spectrum):
    return spectrum / (spectrum.sum() + 1e-12)


def band_fractions(P_norm, freqs):
    out = []
    for c in CENTERS:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        m = (freqs >= lo) & (freqs < hi)
        out.append(P_norm[m].sum())
    return np.array(out)


def centroid_tilt(P_norm, freqs):
    cent = float((freqs * P_norm).sum())
    m = (freqs >= 100) & (freqs <= 10000) & (P_norm > 0)
    tilt = float(np.polyfit(np.log10(freqs[m]), 10 * np.log10(P_norm[m]), 1)[0]) if m.any() else 0.0
    return cent, tilt


def hnr_median(x, sr):
    """Median harmonic-to-noise ratio (dB) over voiced frames via autocorrelation."""
    lag_min, lag_max = int(sr / 500), int(sr / 70)
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


def frame_metrics(P, freqs, x, sr):
    Pn = norm(P)
    total = Pn.sum() + 1e-12

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return Pn[m].sum()

    cent, tilt = centroid_tilt(Pn, freqs)
    naive_p = P / (P.sum() + 1e-12)
    flatness = float(np.exp(np.mean(np.log(naive_p + 1e-12))) / (naive_p.mean() + 1e-12))

    win = np.hanning(FRAME)
    rmss = []
    for i in range(0, max(1, len(x) - FRAME), HOP):
        fr = x[i:i + FRAME]
        if len(fr) < FRAME:
            break
        rmss.append(np.sqrt(np.mean((fr * win) ** 2)))
    rmss = np.array(rmss) if rmss else np.array([1e-12])
    db = 20 * np.log10(rmss + 1e-12)
    active = rmss > 1e-4
    db_act = db[active] if active.any() else db

    return {
        "sr": sr,
        "dur_s": round(len(x) / sr, 3),
        "rms_dbfs": round(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12), 2),
        "peak_dbfs": round(20 * np.log10(np.max(np.abs(x)) + 1e-12), 2),
        "crest_db": round(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)), 2),
        "clip_ratio_pct": round(100 * np.mean(np.abs(x) >= 0.99), 4),
        "dc_offset": round(float(np.mean(x)), 6),
        "noise_floor_dbfs_p5act": round(float(np.percentile(db_act, 5)), 2),
        "dyn_range_db_p95p5act": round(float(np.percentile(db_act, 95) - np.percentile(db_act, 5)), 2),
        "digital_silence_pct": round(100 * float(np.mean(~active)), 2),
        "centroid_hz": round(cent, 1),
        "tilt_db_per_decade": round(tilt, 2),
        "rolloff95_hz": round(float(freqs[np.searchsorted(np.cumsum(Pn) / total, 0.95)]), 1),
        "flatness": round(flatness, 5),
        "hf_8k_16k_pct": round(100 * band(8000, 16000) / total, 2),
        "mid_2k_8k_pct": round(100 * band(2000, 8000) / total, 2),
        "lf_0_2k_pct": round(100 * band(0, 2000) / total, 2),
        "sibilance_6k12k_over_2k6k": round(float(band(6000, 12000) / (band(2000, 6000) + 1e-12)), 4),
        "hnr_db_median": round(hnr_median(x, sr), 2),
    }


def print_metric_table(results):
    names = list(results.keys())
    fields = [k for k in results[names[0]] if k != "tag"]
    print("=== Objective metrics ===")
    print(f"{'metric':>28} | " + " | ".join(f"{n:>12}" for n in names))
    for f in fields:
        print(f"{f:>28} | " + " | ".join(f"{results[n][f]:>12}" for n in names))


# --------------------------------------------------------------------------
# mode 1: A/B comparison
# --------------------------------------------------------------------------
def mode_ab(paths, labels=None):
    labels = labels or [os.path.basename(p) for p in paths]
    results, spectra = {}, {}
    for lbl, p in zip(labels, paths):
        if not os.path.exists(p):
            print(f"[skip] missing file: {p}")
            continue
        x, sr = load(p)
        P, freqs = avg_power_spectrum(x, sr)
        m = frame_metrics(P, freqs, x, sr)
        m["tag"] = lbl
        results[lbl] = m
        spectra[lbl] = (P, freqs, x, sr)

    if not results:
        return
    print_metric_table(results)

    keys = list(spectra.keys())
    if len(keys) == 2:
        a, b = keys
        xa, sra = spectra[a][2], spectra[a][3]
        xb = spectra[b][2]
        n = min(len(xa), len(xb))
        diff = xa[:n] - xb[:n]
        print("\n=== Direct difference ===")
        print(f"aligned_len_s={n/sra:.3f}  "
              f"diff_rms_dbfs={20*np.log10(np.sqrt(np.mean(diff**2))+1e-12):.2f}  "
              f"corr={np.corrcoef(xa[:n], xb[:n])[0,1]:.5f}")

        Pa, f = spectra[a][0], spectra[a][1]
        Pb = spectra[b][0]
        print("\n=== Band-wise energy difference (dB, A - B) ===")
        for lo, hi in [(0, 500), (500, 2000), (2000, 4000), (4000, 8000),
                       (8000, 12000), (12000, 16000), (16000, 20000)]:
            m = (f >= lo) & (f < hi)
            ra = 10 * np.log10(Pa[m].sum() + 1e-12)
            rb = 10 * np.log10(Pb[m].sum() + 1e-12)
            print(f"  {lo:>5}-{hi:<5} Hz : A-B = {ra-rb:+6.2f} dB")


# --------------------------------------------------------------------------
# mode 2: timbre vs training set
# --------------------------------------------------------------------------
def mode_timbre(paths, labels=None, train_list=TRAIN_LIST, n_train=80, max_frames=400, seed=1234):
    labels = labels or [os.path.basename(p) for p in paths]
    if not os.path.exists(train_list):
        print(f"train list not found: {train_list}")
        return
    with open(train_list, encoding="utf-8") as f:
        all_paths = [ln.strip() for ln in f if ln.strip()]

    random.seed(seed)
    sample = random.sample(all_paths, min(n_train, len(all_paths)))
    freqs_ref = np.fft.rfftfreq(FRAME, 1 / 44100)
    acc, used = None, 0
    for p in sample:
        if not os.path.exists(p):
            continue
        x, sr = load(p)
        if sr != 44100:
            continue
        P, freqs = avg_power_spectrum(x, sr, max_frames=max_frames)
        if len(P) != len(freqs_ref):
            continue
        acc = norm(P) if acc is None else acc + norm(P)
        used += 1
    if acc is None or used == 0:
        print("failed to build training-set reference")
        return
    train_ltas = acc / used
    train_bf = band_fractions(train_ltas, freqs_ref)
    tcent, ttilt = centroid_tilt(train_ltas, freqs_ref)

    print(f"training-set reference: {used} files  centroid={tcent:.0f} Hz  tilt={ttilt:+.2f} dB/dec")

    out = {}
    for lbl, p in zip(labels, paths):
        if not os.path.exists(p):
            print(f"[skip] missing file: {p}")
            continue
        x, sr = load(p)
        P, freqs = avg_power_spectrum(x, sr)
        Pn = norm(P)
        out[lbl] = (band_fractions(Pn, freqs),) + centroid_tilt(Pn, freqs)

    if not out:
        return
    keys = list(out.keys())
    print("\n=== 1/3-octave band energy (dB rel. total) and diff vs training set ===")
    print(f"{'band':>8} | {'train':>7} | " + " | ".join(f"{k:>8}" for k in keys)
          + " || " + " | ".join(f"d{k:<6}" for k in keys))
    for i, c in enumerate(CENTERS):
        tdb = 10 * np.log10(train_bf[i] + 1e-12)
        cells = " | ".join(f"{10*np.log10(out[k][0][i]+1e-12):>8.2f}" for k in keys)
        diffs = " | ".join(f"{(10*np.log10(out[k][0][i]+1e-12)) - tdb:>+7.2f}" for k in keys)
        print(f"{c:>6}Hz | {tdb:>7.2f} | {cells} || {diffs}")

    print("\n=== Summary ===")
    for k in keys:
        bf, cent, tilt = out[k]
        d = 10 * np.log10(bf + 1e-12) - 10 * np.log10(train_bf + 1e-12)
        dist = float(np.sqrt(np.mean(d ** 2)))
        hf = sum(bf[i] for i, c in enumerate(CENTERS) if c >= 8000) * 100
        print(f"{k:>12}: centroid={cent:>7.0f} Hz  tilt={tilt:>+6.2f} dB/dec  "
              f"HF(>=8k)={hf:>5.2f}%  timbre_dist_vs_train={dist:>5.2f} dB")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Audio quality evaluation for synthesized wavs (ab / timbre modes)")
    ap.add_argument("mode", nargs="?", choices=["ab", "timbre"], default="ab",
                    help="ab=objective A/B comparison (default), timbre=compare vs training-set timbre")
    ap.add_argument("wavs", nargs="*", help="wav files to analyze")
    ap.add_argument("--labels", default=None, help="comma separated display names (default: file names)")
    ap.add_argument("--train-list", default=TRAIN_LIST, help=f"training list (default {TRAIN_LIST})")
    ap.add_argument("--n", type=int, default=80, help="number of sampled training files in timbre mode (default 80)")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    if not args.wavs:
        ap.print_help()
        return
    if args.mode == "timbre":
        mode_timbre(args.wavs, labels, train_list=args.train_list, n_train=args.n)
    else:
        mode_ab(args.wavs, labels)


if __name__ == "__main__":
    main()
