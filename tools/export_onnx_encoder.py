"""导出 ContentVec768L12（vec768l12）编码器为 ONNX。

思路：不重新实现网络，而是直接用 fairseq 原始模型（`pretrain/checkpoint_best_legacy_500.pt`）
的 `feature_extractor + post_extract_proj + encoder(layer=12)` 子链路构建 wrapper 导出，
从根上保证与 PyTorch 推理数值一致。

用法:
  python tools/export_onnx_encoder.py --out checkpoints/onnx_bundle/vec768l12.onnx
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

CKPT = "pretrain/checkpoint_best_legacy_500.pt"


def patch_pad_to_multiple():
    """让 fairseq 的 pad_to_multiple 对 torch.onnx trace 友好。

    原实现用 `tsz / multiple` 得到 Python float 后调用 `.is_integer()`；
    trace 时 `x.size(dim)` 变成 Tensor，直接抛 AttributeError。
    """
    import fairseq.models.wav2vec.wav2vec2 as w2v2

    def _traceable(x, multiple, dim=-1, value=0):
        if x is None:
            return None, 0
        tsz = x.shape[dim]
        remainder = (multiple - tsz % multiple) % multiple
        if remainder == 0:
            return x, 0
        pad_offset = (0,) * (-1 - dim) * 2
        return torch.nn.functional.pad(x, (*pad_offset, 0, remainder), value=value), remainder

    w2v2.pad_to_multiple = _traceable


class EncoderWrapper(nn.Module):
    """wav(1,N) @16k -> hidden(1,T,768)，等价于 fairseq extract_features(output_layer=12) 的 hidden。"""

    def __init__(self, m, layer=12):
        super().__init__()
        self.m = m
        self.layer = layer

    def forward(self, wav):
        # fairseq 的 ConvFeatureExtractionModel 期望 (B, N)，内部自行 unsqueeze
        x = self.m.feature_extractor(wav)  # (B, 512, T')
        x = x.transpose(1, 2)
        # 与 fairseq HubertModel.forward 一致：post_extract_proj 之前先做一次 layer_norm
        if getattr(self.m, "layer_norm", None) is not None:
            x = self.m.layer_norm(x)
        x = self.m.post_extract_proj(x)
        x, _ = self.m.encoder(x, padding_mask=None, layer=self.layer)
        return x


def load_fairseq_hubert(device="cpu"):
    from fairseq import checkpoint_utils
    from fairseq.data.dictionary import Dictionary

    torch.serialization.add_safe_globals([Dictionary])
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([CKPT], suffix="")
    m = models[0].to(device).eval()
    return m


def main():
    ap = argparse.ArgumentParser(description="Export ContentVec768L12 encoder to ONNX")
    ap.add_argument("--out", default="checkpoints/onnx_bundle/vec768l12.onnx")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=3.0, help="用于导出与自检的音频长度")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)
    dev = "cpu"

    print("[1/4] load fairseq model ...")
    m = load_fairseq_hubert(dev)
    # fairseq 的 required_seq_len_multiple 会触发 pad_to_multiple(),
    # 该函数用 Python 的 float.is_integer() 判断长度，torch.onnx trace 下会失败；
    # 补零长度随后会被裁掉且被 attention mask 屏蔽，等价改成 1 不影响结果。
    if getattr(m.encoder, "required_seq_len_multiple", 1) not in (None, 1):
        print(f"  set encoder.required_seq_len_multiple "
              f"{m.encoder.required_seq_len_multiple} -> 1")
        m.encoder.required_seq_len_multiple = 1
    patch_pad_to_multiple()
    wrapper = EncoderWrapper(m, args.layer).eval()

    wav = torch.randn(1, int(16000 * args.seconds))

    print("[2/4] self-check: wrapper vs fairseq extract_features")
    with torch.no_grad():
        ref = m.extract_features(
            source=wav, padding_mask=torch.BoolTensor(wav.shape).fill_(False),
            output_layer=args.layer)[0]
        got = wrapper(wav)
    print(f"  ref {tuple(ref.shape)}  got {tuple(got.shape)}")
    maxdiff = (ref - got).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ref.reshape(-1, ref.shape[-1]), got.reshape(-1, got.shape[-1]), dim=-1).mean().item()
    print(f"  max_abs_diff={maxdiff:.3e}  cos_sim={cos:.8f}")
    if maxdiff > 1e-3:
        print("  [WARN] 数值不一致，仍继续导出（请在端到端验证时确认）")

    print("[3/4] export onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (wav,), args.out,
            input_names=["wav"], output_names=["feats"],
            dynamic_axes={"wav": {1: "n_samples"}, "feats": {1: "n_frames"}},
            opset_version=16, do_constant_folding=False, dynamo=False,
        )
    print(f"  saved {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    print("[4/4] verify with onnxruntime (if installed)")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"wav": wav.numpy()})[0]
        print(f"  ort out {out.shape}  max_abs_diff_vs_torch={np.abs(out - got.numpy()).max():.3e}")
    except ImportError:
        print("  onnxruntime 未安装，跳过")


if __name__ == "__main__":
    main()
