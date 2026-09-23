"""
TensorBoard / 训练曲线分析工具（统一入口）
========================================
支持四类数据源，按需选择：

1) 离线 tfevents 概览（默认）
   解析 logs/44k 下的 tfevents 事件文件, 输出各标量的最新值、前后对比与趋势,
   用于判断训练是否正常(收敛/波动/NaN/爆炸)。

2) 运行中的 TensorBoard（HTTP）
   从 http://127.0.0.1:6006 之类的服务端拉取标量曲线, 输出首/末/极值与分段均值,
   适合训练进行中快速查看（本地或远程主机均可）。

3) 候选 checkpoint 指标对比
   给定若干训练 step, 输出这些 step 上的关键指标快照, 用于挑选保留哪一版模型。

4) 浅层扩散训练日志解析
   解析 diffusion/log_info.txt, 输出训练/验证 loss 趋势与候选保存点均值。

用法:
  python tools/analyze_tb.py                                   # 离线分析 logs/44k
  python tools/analyze_tb.py logs/44k                          # 离线分析指定目录
  python tools/analyze_tb.py --http http://127.0.0.1:6006 --run .
  python tools/analyze_tb.py --http http://127.0.0.1:6006 --run . --tags loss/g/mel loss/g/fm
  python tools/analyze_tb.py --http http://127.0.0.1:6006 --run . --candidates 68800,75200,77600
  python tools/analyze_tb.py --diffusion-log logs/44k/diffusion/log_info.txt
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.request

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_LOG_DIR = "logs/44k"
DEFAULT_HTTP = "http://127.0.0.1:6006"
DEFAULT_DIFFUSION_LOG = os.path.join("logs", "44k", "diffusion", "log_info.txt")

MAIN_TAGS = ["loss/g/total", "loss/d/total", "loss/g/fm", "loss/g/mel",
             "loss/g/kl", "loss/g/lf0", "learning_rate", "grad_norm_g", "grad_norm_d"]
CANDIDATE_TAGS = ["loss/g/total", "loss/g/mel", "loss/g/kl", "loss/g/fm",
                  "loss/g/lf0", "loss/d/total", "grad_norm_g"]
DIFF_TAGS = ["train/loss", "validation/loss", "train/lr"]

BUCKET_STEPS = 5000  # HTTP 曲线的分段聚合粒度


# --------------------------------------------------------------------------
# 1) 离线 tfevents
# --------------------------------------------------------------------------
def analyze_events(logdir):
    from tensorboard.backend.event_processing import event_accumulator

    files = sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))
    if not files:
        print(f"未找到 tfevents 文件: {logdir}")
        return
    print(f"logdir: {logdir}")
    print(f"tfevents 文件数: {len(files)} (最早的: {os.path.basename(files[0])}, "
          f"最新的: {os.path.basename(files[-1])})")

    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},  # 不截断, 加载全部
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print(f"标量 tag 数: {len(tags)}\n")

    # 优先关注的关键指标
    priority = [t for t in tags if any(k in t.lower() for k in
                ["g/loss", "g/mel", "g/kl", "g/fm", "d/loss", "loss", "perplexity"])]

    rows = []
    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        vals = np.array([e.value for e in events])
        rows.append({
            "tag": tag, "n": len(vals), "start_step": steps[0],
            "end_step": steps[-1], "first": vals[0], "last": vals[-1],
            "min": vals.min(), "max": vals.max(), "mean": vals.mean(),
            "std": vals.std(), "nan": int(np.isnan(vals).sum()),
            "inf": int(np.isinf(vals).sum()),
        })
    rows.sort(key=lambda r: -r["n"])

    print("=" * 110)
    print(f"{'Tag':<38}{'点数':>6}{'范围步':>12}{'首值':>10}{'末值':>10}{'最小':>10}{'最大':>10}{'均值':>10}{'NaN':>5}{'Inf':>5}")
    print("=" * 110)
    for r in rows:
        step_range = f"{r['start_step']}-{r['end_step']}"
        print(f"{r['tag']:<38}{r['n']:>6}{step_range:>12}"
              f"{r['first']:>10.4f}{r['last']:>10.4f}{r['min']:>10.4f}"
              f"{r['max']:>10.4f}{r['mean']:>10.4f}{r['nan']:>5}{r['inf']:>5}")

    # 关键指标趋势 (最后 25 个点)
    print("\n" + "=" * 110)
    print("关键指标最近趋势 (每点=最新 25 个记录)")
    print("=" * 110)
    for tag in priority:
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue
        tail = events[-25:]
        print(f"\n  [{tag}]  共{len(events)}点")
        line = "    "
        for e in tail:
            line += f"{e.value:.3f} "
        print(line)
        first5 = [e.value for e in events[:5]]
        last5 = [e.value for e in events[-5:]]
        print(f"    前5均值={np.mean(first5):.4f}  末5均值={np.mean(last5):.4f}  "
              f"变化={np.mean(last5) - np.mean(first5):+.4f}")


# --------------------------------------------------------------------------
# 2) HTTP 拉取运行中的 TensorBoard
# --------------------------------------------------------------------------
def _fetch_scalars(host, run, tag, timeout=60):
    """从 TensorBoard HTTP API 拉取单个标量序列, 返回按 step 排序的 (step, value) 列表。"""
    url = (f"{host}/data/plugin/scalars/scalars?run="
           f"{urllib.request.quote(run, safe='')}&tag={urllib.request.quote(tag, safe='')}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = json.loads(r.read().decode("utf-8"))
    return sorted(((int(pt[1]), float(pt[2])) for pt in raw), key=lambda p: p[0])


def summarize_http(host, run, tag):
    pts = _fetch_scalars(host, run, tag)
    if not pts:
        print(f"\n=== {tag} ===\n<empty>")
        return
    steps = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    n = len(pts)
    interval = steps[1] - steps[0] if n > 1 else "n/a"
    print(f"\n=== {tag} ===")
    print(f"points={n}  first_step={steps[0]}  last_step={steps[-1]}  (step interval ~{interval})")
    print(f"first={vals[0]:.5f}  last={vals[-1]:.5f}  "
          f"min={min(vals):.5f}@step{steps[vals.index(min(vals))]}  "
          f"max={max(vals):.5f}@step{steps[vals.index(max(vals))]}")
    lo = max(0, n - 50)
    print(f"tail50_avg={sum(vals[lo:]) / (n - lo):.5f}")
    buckets = {}
    for s, v in pts:
        buckets.setdefault(s // BUCKET_STEPS, []).append(v)
    desc = " | ".join(f"{sum(buckets[k]) / len(buckets[k]):.4f}" for k in sorted(buckets))
    print(f"avg per {BUCKET_STEPS}-step bucket: {desc}")


def analyze_http(host, run, tags):
    print(f"host={host}  run={run}")
    for t in tags:
        try:
            summarize_http(host, run, t)
        except Exception as e:  # noqa
            print(f"\n=== {t} ===\nERROR {e}")


# --------------------------------------------------------------------------
# 3) 候选 checkpoint 指标对比
# --------------------------------------------------------------------------
def compare_candidates(host, run, steps, tags=None):
    tags = tags or CANDIDATE_TAGS
    print(f"host={host}  run={run}")
    print(f"候选步: {steps}")
    series = {}
    for t in tags:
        try:
            series[t] = dict(_fetch_scalars(host, run, t))
        except Exception as e:  # noqa
            print(f"[skip] {t}: {e}")
    if not series:
        return
    avail = [t for t in tags if t in series]
    short = [t.split("/")[-1] for t in avail]
    header = f"{'step':>8} | " + " | ".join(f"{s:>10}" for s in short)
    print("\n" + header)
    print("-" * len(header))
    for c in steps:
        row = []
        for t in avail:
            row.append(f"{series[t].get(c, float('nan')):>10.3f}")
        print(f"{c:>8} | " + " | ".join(row))


# --------------------------------------------------------------------------
# 4) 浅层扩散日志解析
# --------------------------------------------------------------------------
def analyze_diffusion_log(path=DEFAULT_DIFFUSION_LOG, ckpt_interval=10000, val_bucket=20000):
    if not os.path.exists(path):
        print(f"未找到扩散日志: {path}")
        return
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    steps, losses, val_steps, val_losses = [], [], [], []
    last_step = None
    for ln in lines:
        m = re.search(r"step: (\d+)\s*$", ln)
        if m and "loss:" in ln:
            lm = re.search(r"loss: ([\d.]+)", ln)
            if lm:
                last_step = int(m.group(1))
                steps.append(last_step)
                losses.append(float(lm.group(1)))
        elif ln.strip().startswith("loss:") and "step" not in ln:
            vm = re.search(r"loss:\s*(\d+(?:\.\d+)?)", ln)
            if vm and last_step is not None:
                val_steps.append(last_step)
                val_losses.append(float(vm.group(1)))

    if not steps:
        print("未解析到训练记录")
        return

    print(f"log: {path}")
    print(f"parsed train points={len(steps)}  val points={len(val_losses)}")
    print(f"train first/last: step {steps[0]} loss {losses[0]:.4f} -> "
          f"step {steps[-1]} loss {losses[-1]:.4f}")

    print("\n=== train loss trend (avg per 5000 steps) ===")
    segs = {}
    for s, l in zip(steps, losses):
        segs.setdefault(s // 5000, []).append(l)
    for k in sorted(segs):
        v = segs[k]
        print(f"step {k*5000:>7}-{k*5000+4999:>7}: n={len(v):3d}  "
              f"avg={sum(v)/len(v):.4f}  min={min(v):.4f}  max={max(v):.4f}")

    if val_losses:
        print(f"\n=== validation loss trend (avg per {val_bucket} steps) ===")
        vs = {}
        for s, l in zip(val_steps, val_losses):
            vs.setdefault(s // val_bucket, []).append(l)
        for k in sorted(vs):
            v = vs[k]
            print(f"step {k*val_bucket:>7}-{k*val_bucket+val_bucket-1:>7}: n={len(v):2d}  "
                  f"avg={sum(v)/len(v):.4f}  min={min(v):.4f}  max={max(v):.4f}")
        best = min(zip(val_losses, val_steps))
        print(f"\nbest validation: {best[0]:.4f} at step ~{best[1]}")

    # 候选保存点邻近均值
    print(f"\n=== candidate checkpoint nearby train loss (interval={ckpt_interval}) ===")
    ckpts = list(range(ckpt_interval, steps[-1] + 1, ckpt_interval))
    if steps[-1] not in ckpts:
        ckpts.append(steps[-1])
    idx = 0
    for c in ckpts:
        while idx < len(steps) and steps[idx] <= c:
            idx += 1
        lo = max(0, idx - 100)
        seg = losses[lo:idx]
        if seg:
            print(f"ckpt {c:>7}: avg_last100={sum(seg)/len(seg):.4f} (from step {steps[lo]})")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TensorBoard / 训练曲线分析工具（统一入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("logdir", nargs="?", default=None,
                    help=f"离线 tfevents 目录（默认 {DEFAULT_LOG_DIR}）")
    ap.add_argument("--http", default=None, help=f"TensorBoard 服务地址，如 {DEFAULT_HTTP}")
    ap.add_argument("--run", default=".", help="run 名称（默认 '.'）")
    ap.add_argument("--tags", nargs="*", default=None, help="要分析的标量 tag 列表")
    ap.add_argument("--candidates", default=None,
                    help="逗号分隔的 step 列表，输出候选 checkpoint 指标对比")
    ap.add_argument("--diffusion-log", default=None,
                    help=f"解析扩散训练日志（默认 {DEFAULT_DIFFUSION_LOG}）")
    args = ap.parse_args()

    if args.diffusion_log is not None:
        path = args.diffusion_log or DEFAULT_DIFFUSION_LOG
        analyze_diffusion_log(path)
        return

    if args.http is not None or args.candidates is not None:
        host = args.http or DEFAULT_HTTP
        if args.candidates:
            steps = [int(s) for s in args.candidates.split(",") if s.strip()]
            compare_candidates(host, args.run, steps, args.tags)
        else:
            tags = args.tags or (DIFF_TAGS if "diffusion" in args.run else MAIN_TAGS)
            analyze_http(host, args.run, tags)
        return

    analyze_events(args.logdir or DEFAULT_LOG_DIR)


if __name__ == "__main__":
    main()
