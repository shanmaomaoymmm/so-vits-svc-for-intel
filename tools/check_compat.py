"""
与上游 so-vits-svc 项目的兼容性检查工具
======================================
回答两个问题：
  1) 当前 config.json 能否被上游项目读取（上游代码引用的 hps 字段是否齐全）
  2) 当前 G / D 检查点能否被上游项目的模型结构加载（state_dict 键名是否匹配）

原理：
  - 字段检查：扫描上游 *.py 中 `hps.<section>.<field>` 的引用，与本地 config 对比
  - 权重检查：把上游目录加入 sys.path 首位后动态加载上游 models.py，
    实例化对应模型并比较 state_dict 键名集合（不加载真实权重，避免设备依赖）

用法:
  python tools/check_compat.py
  python tools/check_compat.py --upstream "C:/Users/Qisato/Desktop/Code/so-vits-svc" \
      --config logs/44k/config.json --g logs/44k/G_100000.pth --d logs/44k/D_100000.pth

注: 输出文本使用英文以避免 Windows 终端编码问题。
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

import torch

DEFAULT_UPSTREAM = r"C:\Users\Qisato\Desktop\Code\so-vits-svc"
DEFAULT_CONFIG = os.path.join("logs", "44k", "config.json")
HPS_REF = re.compile(r"hps\.(train|model|data|spk)\.(\w+)")
# 这些是 dict 方法调用（如 hps.spk.keys()），不是配置字段，需排除
METHOD_NAMES = {"keys", "items", "values", "get", "update", "copy", "pop"}


def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_upstream_fields(upstream_dir):
    """Return the set of hps.<section>.<field> referenced by upstream *.py files."""
    refs = set()
    # 递归扫描全部 .py（含 inference/ diffusion/ vencoder/ vdecoder/ 等推理链路）
    for p in glob.glob(os.path.join(upstream_dir, "**", "*.py"), recursive=True):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                for m in HPS_REF.finditer(f.read()):
                    if m.group(2) in METHOD_NAMES:
                        continue
                    refs.add(f"{m.group(1)}.{m.group(2)}")
        except Exception:
            continue
    return refs


def check_config(upstream_dir, config_path):
    refs = collect_upstream_fields(upstream_dir)
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    missing = []
    for ref in sorted(refs):
        section, field = ref.split(".", 1)
        if section not in cfg or field not in cfg.get(section, {}):
            missing.append(ref)

    print("=== 1) config field compatibility ===")
    print(f"upstream references {len(refs)} hps fields")
    if missing:
        print(f"MISSING ({len(missing)}): {missing}")
    else:
        print("OK: all upstream-referenced fields exist in local config")

    extra = []
    for section in ("train", "model", "data"):
        for k in cfg.get(section, {}):
            if f"{section}.{k}" not in refs:
                extra.append(f"{section}.{k}")
    print(f"local-only fields (ignored by upstream): {len(extra)}")
    return missing


def compare_keys(label, upstream_keys, local_keys):
    only_up = sorted(upstream_keys - local_keys)
    only_local = sorted(local_keys - upstream_keys)
    common = len(upstream_keys & local_keys)
    print(f"\n--- {label} ---")
    print(f"upstream model keys={len(upstream_keys)}  local ckpt keys={len(local_keys)}  common={common}")
    if not only_up and not only_local:
        print("COMPATIBLE: key sets are identical")
    elif only_up and not only_local:
        print(f"MISSING in local ckpt ({len(only_up)}): {only_up[:8]}")
    elif only_local and not only_up:
        print(f"EXTRA in local ckpt ({len(only_local)}): {only_local[:8]}")
    else:
        print(f"MISMATCH: upstream-only={len(only_up)} e.g. {only_up[:4]}")
        print(f"          local-only={len(only_local)} e.g. {only_local[:4]}")
    return only_up, only_local


def main():
    ap = argparse.ArgumentParser(description="Compatibility check against upstream so-vits-svc")
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="upstream project directory")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="local config.json to test")
    ap.add_argument("--g", default=None, help="local generator checkpoint (default: newest logs/44k/G_*.pth)")
    ap.add_argument("--d", default=None, help="local discriminator checkpoint (default: newest logs/44k/D_*.pth)")
    args = ap.parse_args()

    if not os.path.isdir(args.upstream):
        print(f"upstream dir not found: {args.upstream}")
        return
    print(f"upstream : {args.upstream}")
    print(f"config   : {args.config}")

    check_config(args.upstream, args.config)

    # ---- import upstream models with its own modules package ----
    sys.path.insert(0, args.upstream)
    try:
        orig = load_module_from_path(os.path.join(args.upstream, "models.py"), "upstream_models")
        print("\n[ok] upstream models.py imported")
    except Exception as e:  # noqa
        print(f"\n[fail] cannot import upstream models.py: {e}")
        return

    with open(args.config, encoding="utf-8") as f:
        hps = json.load(f)

    g_path = args.g or max(glob.glob(os.path.join("logs", "44k", "G_*.pth")),
                           key=lambda p: int(re.search(r"_(\d+)\.pth$", p).group(1)), default=None)
    d_path = args.d or max(glob.glob(os.path.join("logs", "44k", "D_*.pth")),
                           key=lambda p: int(re.search(r"_(\d+)\.pth$", p).group(1)), default=None)
    print(f"\nG ckpt   : {g_path}")
    print(f"D ckpt   : {d_path}")

    # ---- G ----
    try:
        net_g = orig.SynthesizerTrn(
            hps["data"]["filter_length"] // 2 + 1,
            hps["train"]["segment_size"] // hps["data"]["hop_length"],
            **hps["model"])
        up_g = set(net_g.state_dict().keys())
    except Exception as e:  # noqa
        print(f"\n[fail] cannot build upstream SynthesizerTrn with local config: {e}")
        up_g = set()
    if up_g and g_path and os.path.exists(g_path):
        ck = torch.load(g_path, map_location="cpu", weights_only=False)
        compare_keys("Generator (SynthesizerTrn)", up_g, set(ck["model"].keys()))
        # 严格加载测试：验证上游模型能否直接 load_state_dict
        try:
            missing, unexpected = net_g.load_state_dict(ck["model"], strict=False)
            ok = (not missing) and (not unexpected)
            print(f"strict load: missing={len(missing)} unexpected={len(unexpected)} -> "
                  f"{'OK (upstream can load this G checkpoint)' if ok else 'MISMATCH'}")
        except Exception as e:  # noqa
            print(f"strict load FAILED: {e}")

    # ---- D ----
    if d_path and os.path.exists(d_path):
        dck = torch.load(d_path, map_location="cpu", weights_only=False)
        local_d = set(dck["model"].keys())
        print(f"\nlocal D key prefixes: {sorted({k.split('.')[0] for k in local_d})}")
        try:
            net_d = orig.MultiPeriodDiscriminator()
            up_d = set(net_d.state_dict().keys())
            print(f"upstream D key prefixes: {sorted({k.split('.')[0] for k in up_d})}")
            compare_keys("Discriminator", up_d, local_d)
        except Exception as e:  # noqa
            print(f"[warn] cannot build upstream discriminator: {e}")


if __name__ == "__main__":
    main()
