+"""ONNX 推理包 vs PyTorch 逐级数值对齐验证。

逐组件比较（相同输入），定位任何不一致：
  1) RMVPE  f0 / uv
  2) ContentVec768L12  c
  3) Volume_Extractor  vol
  4) SoVits 输出波形
  5) 扩散组件 condition / denoise / pred / after
  6) NSF-HiFiGAN 波形

用法:
  python tools/verify_onnx_bundle.py --wav raw/test_eval.wav --bundle checkpoints/onnx_bundle
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("checkpoints", "onnx_bundle"))

import infer as oi  # noqa: E402
import utils  # noqa: E402
from inference.infer_tool import Svc  # noqa: E402

MODEL = "logs/44k/G_100000.pth"
CONFIG = "logs/44k/config.json"


def report(name, a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        shp = f"shape {a.shape} vs {b.shape} "
        n = min(a.size, b.size)
    else:
        shp = ""
        n = a.size
    ra, rb = a.reshape(-1)[:n], b.reshape(-1)[:n]
    d = float(np.abs(ra - rb).max())
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
    cos = float(np.dot(ra, rb) / denom) if denom > 0 else float("nan")
    print(f"  {name:22s} {shp}max_abs_diff={d:.3e}  cos={cos:.6f}")
    return d, cos


def report_mel(name, a, b, svc_o):
    """波形无法逐点对齐时，比较 log-mel 域（对随机噪声不敏感）。"""
    mel_a = svc_o.nsf_log_mel(np.asarray(a, dtype=np.float32))
    mel_b = svc_o.nsf_log_mel(np.asarray(b, dtype=np.float32))
    n = min(mel_a.shape[0], mel_b.shape[0])
    d = np.abs(mel_a[:n] - mel_b[:n])
    print(f"  {name:22s} log-mel L1={d.mean():.3f} dB  max={d.max():.3f} dB")
    rms_a = 20 * np.log10(np.sqrt(np.mean(np.asarray(a) ** 2)) + 1e-12)
    rms_b = 20 * np.log10(np.sqrt(np.mean(np.asarray(b) ** 2)) + 1e-12)
    print(f"  {name:22s} rms {rms_a:.2f} dBFS vs {rms_b:.2f} dBFS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="raw/test_eval.wav")
    ap.add_argument("--bundle", default=os.path.join("checkpoints", "onnx_bundle"))
    args = ap.parse_args()

    wav, sr = sf.read(args.wav, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    assert sr == 44100, sr
    p_len = len(wav) // 512
    print(f"wav {args.wav}  {len(wav)/sr:.2f}s  p_len={p_len}")

    print("\n[setup] PyTorch Svc (cpu) ...")
    svc = Svc(MODEL, CONFIG, "cpu", "", False,
              "logs/44k/diffusion/model_188000.pt", "logs/44k/diffusion/config.yaml",
              True, False, False, False, vocoder_device="cpu")
    o = oi.OnnxSvc(bundle=args.bundle, shallow_diffusion=True)

    # ------------------------------------------------------------------ 1) f0
    print("\n[1] RMVPE f0 / uv")
    f0pred = utils.get_f0_predictor("rmvpe", hop_length=512, sampling_rate=44100,
                                    device="cpu", threshold=0.05)
    pf0, puv = f0pred.compute_f0_uv(wav, p_len)
    of0_raw, _ = o.rmvpe_f0(wav)
    of0, ouv = o.post_process_f0(of0_raw, p_len)
    report("f0", pf0, of0)
    report("uv", puv, ouv)

    # ------------------------------------------------------ 2) contentvec c
    print("\n[2] ContentVec768L12 c (以 ONNX 的 16k 输入对齐)")
    w16 = torch.from_numpy(o.to16k(wav, width=6))          # 1D，hubert_encode 内部会 view(1,-1)
    with torch.no_grad():
        pc = svc.hubert_model.encoder(w16)                      # (1,768,T)
    pc = utils.repeat_expand_2d(pc.squeeze(0), p_len, "nearest")
    oc = o.contentvec(wav, p_len)                               # (1,T,768)
    report("c", pc.numpy(), oc[0].T)

    # ---------------------------------------------------------- 3) volume
    print("\n[3] Volume_Extractor")
    pv = utils.Volume_Extractor(512).extract(torch.FloatTensor(wav)[None, :]).numpy()
    ov = oi.volume_extract(wav, 512)
    report("vol", pv, ov)

    # ---------------------------------------------------------- 4) SoVits
    print("\n[4] SoVits 输出波形（相同 c/f0/uv/vol/noise）")
    # net_g_ms.infer 内部会 torch.manual_seed(52468) 后 randn_like 采样，
    # 这里用同一 seed 复现该噪声并注入 ONNX（两者使用同一随机源才可比）。
    torch.manual_seed(52468)
    noise_pt = torch.randn(1, 192, p_len)
    noise = (noise_pt * 0.4).numpy().astype(np.float32)
    sid = torch.LongTensor([0]).unsqueeze(0)
    c_pt = np.ascontiguousarray(oc[0].T)[None].astype(np.float32)      # (1,768,T)
    with torch.no_grad():
        pt_audio, _ = svc.net_g_ms.infer(
            torch.from_numpy(c_pt).float(),
            f0=torch.from_numpy(of0[None]).float(),
            g=sid,
            uv=torch.from_numpy(ouv[None]).float(),
            predict_f0=False, noice_scale=0.4,
            vol=torch.from_numpy(ov[None]).float())
    pt_audio = pt_audio[0, 0].cpu().numpy()
    inputs = {
        "c": oc.astype(np.float32),
        "f0": of0[None].astype(np.float32),
        "mel2ph": np.arange(p_len, dtype=np.int64)[None],
        "uv": ouv[None].astype(np.float32),
        "noise": noise,
        "sid": np.array([0], dtype=np.int64),
        "vol": ov[None].astype(np.float32),
    }
    onx_audio = np.asarray(o.s_sovits.run(None, inputs)[0]).reshape(-1)
    # dec 内部的激励噪声在 ONNX 中被外置复用同一 noise 输入（与 PyTorch 内部独立随机不同源），
    # 故波形不可逐点对齐 -> 改在 log-mel 域比较。
    report_mel("audio(noice_scale=0.4)", pt_audio, onx_audio, o)

    # noice_scale=1.0：ONNX 侧直接传未缩放噪声
    with torch.no_grad():
        pt_audio1, _ = svc.net_g_ms.infer(
            torch.from_numpy(c_pt).float(),
            f0=torch.from_numpy(of0[None]).float(), g=sid,
            uv=torch.from_numpy(ouv[None]).float(), predict_f0=False,
            noice_scale=1.0, vol=torch.from_numpy(ov[None]).float())
    inputs1 = dict(inputs, noise=noise_pt.numpy().astype(np.float32))
    onx_audio1 = np.asarray(o.s_sovits.run(None, inputs1)[0]).reshape(-1)
    report_mel("audio(noice_scale=1.0)", pt_audio1[0, 0].cpu().numpy(), onx_audio1, o)

    # ------------------------------------------------- 5) diffusion 组件
    print("\n[5] 扩散组件（condition / denoise / pred / after）")
    mel_np = o.nsf_log_mel(onx_audio)                            # (T,128)
    T = mel_np.shape[0]
    pt_mel = svc.vocoder.extract(torch.from_numpy(onx_audio)[None, :], 44100)[0].numpy()
    report("nsf log-mel", mel_np, pt_mel)

    f0d = of0[:T]
    vol_d = oi.volume_extract(onx_audio, 512)[:T]
    mel2ph = (np.arange(T, dtype=np.int64) + 1)[None]
    cond_onnx = np.asarray(o.s_denc.run(None, {
        "hubert": oc[:, :T].astype(np.float32),
        "mel2ph": mel2ph,
        "f0": f0d[None].astype(np.float32),
        "volume": vol_d[None].astype(np.float32)})[0])
    dec = svc.diffusion_model.decoder
    with torch.no_grad():
        pt_cond = svc.diffusion_model.unit_embed(torch.from_numpy(oc[:, :T]).float()) \
            + svc.diffusion_model.f0_embed(
                torch.log(1 + torch.from_numpy(f0d[None, :, None]).float() / 700)) \
            + svc.diffusion_model.volume_embed(torch.from_numpy(vol_d[None, :, None]).float())
        pt_cond = pt_cond.transpose(1, 2).numpy()
    report("condition", pt_cond, cond_onnx)

    x = np.random.default_rng(1).standard_normal((1, 1, 128, T)).astype(np.float32)
    t = np.array([90], dtype=np.int64)
    npred = np.asarray(o.s_dnoise.run(None, {"noise": x, "time": t,
                                             "condition": cond_onnx})[0])
    with torch.no_grad():
        ptn = dec.denoise_fn(torch.from_numpy(x).float(), torch.from_numpy(t),
                             cond=torch.from_numpy(cond_onnx).float()).numpy()
    report("denoise", ptn, npred)

    t2 = np.array([80], dtype=np.int64)
    xp = np.asarray(o.s_dpred.run(None, {"noise": x, "noise_pred": npred,
                                         "time": t, "time_prev": t2})[0])
    ac = dec.alphas_cumprod.numpy()
    a_t, a_prev = ac[t[0]], ac[t2[0]]
    x_delta = (a_prev - a_t) * ((1 / (np.sqrt(a_t) * (np.sqrt(a_t) + np.sqrt(a_prev)))) * x
                                - 1 / (np.sqrt(a_t) * (np.sqrt((1 - a_prev) * a_t)
                                                      + np.sqrt((1 - a_t) * a_prev))) * npred)
    report("pred(get_x_pred)", x + x_delta, xp)

    xa = np.asarray(o.s_dafter.run(None, {"x": x})[0]).reshape(-1)
    smin = float(dec.spec_min.reshape(-1)[0].numpy())
    smax = float(dec.spec_max.reshape(-1)[0].numpy())
    pta = ((x + 1) / 2 * (smax - smin) + smin).reshape(-1)
    report("after(denorm)", pta, xa)

    # ------------------------------------------------- 6) vocoder
    print("\n[6] NSF-HiFiGAN 声码器（相同 mel/f0/rand_ini/noise）")
    mel_pt = torch.from_numpy(mel_np.T[None]).float()
    f0_v = torch.from_numpy(f0d[None]).float()
    rnd = np.random.default_rng(2)
    rand_ini = rnd.random((1, 9)).astype(np.float32)
    voc_noise = rnd.standard_normal((1, T * 512, 9)).astype(np.float32)

    from tools.export_onnx_vocoder import VocoderWrapper
    gen, _ = _load_gen()
    vw = VocoderWrapper(gen).eval()
    with torch.no_grad():
        pt_v = vw(mel_pt, f0_v, torch.from_numpy(rand_ini), torch.from_numpy(voc_noise)).numpy()
    on_v = np.asarray(o.s_voc.run(None, {"mel": mel_pt.numpy().astype(np.float32),
                                         "f0": f0_v.numpy().astype(np.float32),
                                         "rand_ini": rand_ini, "noise": voc_noise})[0])
    report("vocoder audio", pt_v.reshape(-1), on_v.reshape(-1))

    print("\ndone")


def _load_gen():
    from vdecoder.nsf_hifigan.models import load_model
    return load_model("pretrain/nsf_hifigan/model", device="cpu")


if __name__ == "__main__":
    main()
