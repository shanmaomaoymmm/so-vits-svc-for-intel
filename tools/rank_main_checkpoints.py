"""
主模型候选 checkpoint 排序工具
==============================
从 logs/44k 的 tfevents 中解析逐 step 的关键标量，对**磁盘上实际存在**的
G_*.pth 检查点按滑动窗口均值排序，用于挑选推理/导出用的主模型。

排序依据（同一次训练内可比）:
  - loss/g/mel   (权重 45, 谱重建，越低越好)
  - loss/g/total (综合，越低越好)
  - loss/g/fm    (特征匹配，越低越好)
  - loss/d/total (应接近 platform 平衡点, 过低说明判别器占优)
  - grad_norm_g  (稳定性，越低越稳)

复合分数 = z(mel) + z(total) + 0.5*z(fm)，z 为全序列标准化。

用法:
  python tools/rank_main_checkpoints.py
  python tools/rank_main_checkpoints.py --logdir logs/44k --window 5 --top 8
  python tools/rank_main_checkpoints.py --steps 80000,85000,90000
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TAGS = ["loss/g/total", "loss/g/mel", "loss/g/fm", "loss/g/kl",
        "loss/d/total", "grad_norm_g"]


def load_scalars(logdir):
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        logdir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        pts = ea.Scalars(tag)
        out[tag] = np.array([[e.step, e.value] for e in pts], dtype=np.float64)
    return out


def window_mean(series, step, half):
    """series: Nx2 [step, value]; return mean value within +/- half steps."""
    m = np.abs(series[:, 0] - step) <= half
    return float(series[m, 1].mean()) if m.any() else float("nan")


def list_ckpt_steps(logdir):
    steps = []
    for p in glob.glob(os.path.join(logdir, "G_*.pth")):
        m = re.search(r"_(\d+)\.pth$", p)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def main():
    ap = argparse.ArgumentParser(description="Rank main-model checkpoints by smoothed TB metrics")
    ap.add_argument("--logdir", default=os.path.join("logs", "44k"))
    ap.add_argument("--window", type=int, default=5,
                    help="half window in log points (1 point = log_interval steps)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--steps", default=None,
                    help="comma separated candidate steps (default: detect from G_*.pth)")
    args = ap.parse_args()

    scal = load_scalars(args.logdir)
    missing = [t for t in TAGS if t not in scal]
    if missing:
        print(f"[WARN] missing tags: {missing}")
    tag_ok = [t for t in TAGS if t in scal]

    # z-score per tag for composite
    z = {}
    for t in tag_ok:
        v = scal[t][:, 1]
        z[t] = (v - v.mean()) / (v.std() + 1e-12)

    steps = ([int(s) for s in args.steps.split(",") if s.strip()]
             if args.steps else list_ckpt_steps(args.logdir))
    if not steps:
        print("no candidate steps")
        return

    # uniform half-window in steps (approximate from stored steps of first tag)
    stored = scal[tag_ok[0]][:, 0]
    log_interval = int(stored[1] - stored[0]) if len(stored) > 1 else 200
    half = args.window * log_interval

    rows = []
    for s in steps:
        r = {"step": s}
        for t in tag_ok:
            r[t] = window_mean(scal[t], s, half)
            r[f"z_{t}"] = window_mean(
                np.column_stack([scal[t][:, 0], z[t]]), s, half)
        # composite: prioritize mel + total, mild fm penalty
        comp = 0.0
        for t, w in (("loss/g/mel", 1.0), ("loss/g/total", 1.0), ("loss/g/fm", 0.5)):
            if f"z_{t}" in r and not np.isnan(r[f"z_{t}"]):
                comp += w * r[f"z_{t}"]
        r["score"] = comp
        rows.append(r)

    rows.sort(key=lambda x: x["score"])

    hdr = (f"{'rank':>4} {'step':>8} | {'score':>7} | {'g_total':>8} {'mel':>8} "
           f"{'fm':>7} {'kl':>6} {'d_total':>8} {'grad_g':>9}")
    print(f"logdir={args.logdir}  window=+/-{half} steps  candidates={len(rows)}")
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:>4} {r['step']:>8} | {r['score']:>7.3f} | "
              f"{r.get('loss/g/total', float('nan')):>8.3f} "
              f"{r.get('loss/g/mel', float('nan')):>8.3f} "
              f"{r.get('loss/g/fm', float('nan')):>7.3f} "
              f"{r.get('loss/g/kl', float('nan')):>6.3f} "
              f"{r.get('loss/d/total', float('nan')):>8.3f} "
              f"{r.get('grad_norm_g', float('nan')):>9.1f}")

    print("\n[worse tail]")
    for i, r in enumerate(rows[-3:], len(rows) - 2):
        print(f"{i:>4} {r['step']:>8} | {r['score']:>7.3f} | "
              f"{r.get('loss/g/total', float('nan')):>8.3f} "
              f"{r.get('loss/g/mel', float('nan')):>8.3f} "
              f"{r.get('loss/g/fm', float('nan')):>7.3f}")

    print("\n=== all candidates (sorted) ===")
    line = "  ".join(f"{r['step']}({r['score']:+.2f})" for r in rows)
    print(line)


if __name__ == "__main__":
    main()
