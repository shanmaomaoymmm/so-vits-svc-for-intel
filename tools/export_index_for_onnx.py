"""将特征检索索引 feature_and_index.pkl 导出为 MoeVoiceStudio 可用的 .index 文件。

用法:
  python tools/export_index_for_onnx.py \
      --pkl model/森息花雾6.0/feature_and_index.pkl \
      --out-dir checkpoints/huawu_main_389600
"""
import argparse
import os
import pickle

import faiss


def export(pkl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(pkl_path, "rb") as f:
        indexs = pickle.load(f)
    for k in indexs:
        print(f"Save {k} index")
        faiss.write_index(indexs[k], os.path.join(out_dir, f"Index-{k}.index"))
    print(f"Saved all index -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export faiss index for ONNX deployment")
    ap.add_argument("--pkl", default="feature_and_index.pkl",
                    help="feature_and_index.pkl path")
    ap.add_argument("--out-dir", default="checkpoints/crs",
                    help="output directory for Index-*.index")
    args = ap.parse_args()
    export(args.pkl, args.out_dir)
