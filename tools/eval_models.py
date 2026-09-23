"""
候选模型批量推理评估脚本
========================
对若干 (主模型, 配置) 组合运行 inference_main.py，输出统一命名到 results_eval/，
便于随后用 tools/compare_audio.py 做客观对比。

用法:
  python tools/eval_models.py --case main_100000=logs/44k/G_100000.pth=logs/44k/config.json
  python tools/eval_models.py --raw test_eval.wav \
      --case a=logs/44k/G_98000.pth=logs/44k/config.json \
      --case b=logs/44k/G_96000.pth=logs/44k/config.json
  # 浅层扩散（对同一主模型比较不同扩散 checkpoint）
  python tools/eval_models.py --shd --diffusion logs/44k/diffusion/model_150000.pt \
      --diffusion-config logs/44k/diffusion/config.yaml \
      --case g98_shd150=logs/44k/G_98000.pth=logs/44k/config.json
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT_DIR = "results_eval"
RESULTS_DIR = "results"


def run_case(label, model, config, raw, diffusion=None, diffusion_config=None,
             k_step=100, shd=False, f0p="rmvpe", vocoder_device="cpu"):
    os.makedirs(OUT_DIR, exist_ok=True)
    before = set(glob.glob(os.path.join(RESULTS_DIR, "*")))
    cmd = [sys.executable, "inference_main.py",
           "-m", model, "-c", config, "-n", raw, "-s", "huawu",
           "-f0p", f0p, "-vd", vocoder_device, "-wf", "wav"]
    if shd:
        cmd += ["-shd", "-ks", str(k_step),
                "-dm", diffusion, "-dc", diffusion_config]
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[FAIL] {label} returncode={r.returncode}")
        return False
    after = set(glob.glob(os.path.join(RESULTS_DIR, "*")))
    new = sorted(after - before, key=os.path.getmtime)
    if not new:
        print(f"[FAIL] {label} produced no output")
        return False
    dst = os.path.join(OUT_DIR, f"{label}.wav")
    shutil.move(new[-1], dst)
    print(f"[OK] {label} -> {dst}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Batch inference for model candidates")
    ap.add_argument("--case", action="append", required=True,
                    help="label=model_path=config_path (repeatable)")
    ap.add_argument("--raw", default="test_eval.wav", help="input wav under raw/")
    ap.add_argument("--diffusion", default=None)
    ap.add_argument("--diffusion-config", default=None)
    ap.add_argument("--k-step", type=int, default=100)
    ap.add_argument("--shd", action="store_true", help="enable shallow diffusion")
    ap.add_argument("--f0p", default="rmvpe")
    ap.add_argument("--vocoder-device", default="cpu")
    args = ap.parse_args()

    ok = 0
    for spec in args.case:
        parts = spec.split("=")
        if len(parts) != 3:
            print(f"[skip] bad case spec: {spec}")
            continue
        label, model, config = parts
        if run_case(label, model, config, args.raw, args.diffusion,
                    args.diffusion_config, args.k_step, args.shd,
                    args.f0p, args.vocoder_device):
            ok += 1
    print(f"\ncompleted {ok}/{len(args.case)} cases")


if __name__ == "__main__":
    main()
