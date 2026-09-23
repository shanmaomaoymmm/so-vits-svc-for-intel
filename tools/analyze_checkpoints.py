"""
Checkpoint 体积与磁盘占用分析工具
=================================
测量主模型 (G/D) 与浅层扩散 checkpoint 的实际体积构成（权重 vs 优化器状态），
并按不同的保存间隔预估训练到目标步数所需的磁盘空间。

背景: 主模型 checkpoint 通过 utils.save_checkpoint() 会一并写入 optimizer.state_dict()，
      优化器状态通常占单份文件的 ~2/3；若只为选模/发布（不需断点续训），可只保留权重以省空间。

用法:
  python tools/analyze_checkpoints.py
  python tools/analyze_checkpoints.py --target-steps 100000 --intervals 1000,2000,4000,8000
  python tools/analyze_checkpoints.py --g logs/44k/G_126400.pth --d logs/44k/D_126400.pth --diffusion logs/44k/diffusion/model_188000.pt

注: 输出文本使用英文以避免 Windows 终端编码导致的乱码。
"""
import argparse
import glob
import os
import re


def find_latest(pattern):
    """Pick the file with the largest step number from a glob pattern."""
    files = glob.glob(pattern)
    if not files:
        return None

    def step_of(p):
        m = re.search(r"_(\d+)\.(pth|pt)$", p)
        return int(m.group(1)) if m else -1

    return max(files, key=step_of)


def _tensor_bytes(state):
    total = 0
    for v in state.values():
        if hasattr(v, "numel"):
            total += v.numel() * v.element_size()
    return total


def analyze_file(path):
    """Return (on-disk bytes, weight bytes, optimizer bytes)."""
    import torch

    disk = os.path.getsize(path) if os.path.exists(path) else 0
    msize = osize = 0
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        model = d.get("model", d) if isinstance(d, dict) else d
        if isinstance(model, dict):
            msize = _tensor_bytes(model)
        opt = d.get("optimizer") if isinstance(d, dict) else None
        if isinstance(opt, dict) and "state" in opt:
            for st in opt["state"].values():
                if isinstance(st, dict):
                    osize += _tensor_bytes(st)
    except Exception as e:  # noqa
        print(f"[WARN] failed to parse {path}: {e}")
    return disk, msize, osize


def mb(x):
    return x / 1024 / 1024


def main():
    ap = argparse.ArgumentParser(description="Checkpoint size / disk usage analyzer")
    ap.add_argument("--g", default=None,
                    help="generator checkpoint (default: newest logs/44k/G_*.pth)")
    ap.add_argument("--d", default=None,
                    help="discriminator checkpoint (default: newest logs/44k/D_*.pth)")
    ap.add_argument("--diffusion", default=None,
                    help="diffusion checkpoint (default: newest logs/44k/diffusion/model_*.pt)")
    ap.add_argument("--target-steps", type=int, default=100000,
                    help="target training steps for the projection (default 100000)")
    ap.add_argument("--intervals", default="1000,2000,4000,8000",
                    help="comma separated save intervals for the projection")
    args = ap.parse_args()

    g_path = args.g or find_latest(os.path.join("logs", "44k", "G_*.pth"))
    d_path = args.d or find_latest(os.path.join("logs", "44k", "D_*.pth"))
    diff_path = args.diffusion or find_latest(os.path.join("logs", "44k", "diffusion", "model_*.pt"))

    print("=== auto-selected checkpoints ===")
    print(f"  G        : {g_path}")
    print(f"  D        : {d_path}")
    print(f"  diffusion: {diff_path}")

    sizes, detail = {}, {}
    print("\n=== on-disk size / internal composition ===")
    for name, p in (("G", g_path), ("D", d_path), ("diffusion", diff_path)):
        if not p or not os.path.exists(p):
            print(f"{name:>9}: <not found>")
            continue
        disk, m, o = analyze_file(p)
        sizes[name], detail[name] = disk, (m, o)
        print(f"{name:>9}: disk={mb(disk):8.2f} MB  weights={mb(m):8.2f} MB  "
              f"optimizer={mb(o):8.2f} MB  ({p})")

    if "G" in sizes and "D" in sizes:
        full = sizes["G"] + sizes["D"]
        pure = detail["G"][0] + detail["D"][0]
        print("\n=== per-save cost (main model G+D) ===")
        print(f"  full (with optimizer) : {mb(full):8.2f} MB")
        print(f"  weights-only          : {mb(pure):8.2f} MB")
        if "diffusion" in sizes:
            print(f"  diffusion (weights)   : {mb(sizes['diffusion']):8.2f} MB")

        intervals = [int(x) for x in args.intervals.split(",") if x.strip()]
        print(f"\n=== projected disk usage (training to {args.target_steps} steps) ===")
        print(f"{'interval':>9} | {'count':>6} | {'full(with opt)':>16} | {'weights-only':>14}")
        for iv in intervals:
            cnt = args.target_steps // iv + 1
            print(f"{iv:>9} | {cnt:>6} | {mb(cnt*full)/1024:>13.1f} GB | {mb(cnt*pure)/1024:>11.1f} GB")
        if "diffusion" in sizes:
            print("\n  [diffusion, weights-only]")
            for iv in intervals:
                if iv < 5000:
                    continue
                cnt = args.target_steps // iv + 1
                print(f"    interval={iv:>6}: {cnt:>4} files ~= {mb(cnt*sizes['diffusion'])/1024:>5.1f} GB")


if __name__ == "__main__":
    main()
