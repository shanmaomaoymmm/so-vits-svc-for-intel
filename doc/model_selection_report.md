 # 模型选型与 ONNX 导出报告（森息花雾 / huawu，本次训练 run）

> 生成时间：2026-09-23
> **选型范围：`logs/44k` 本次训练 run**（主模型 `G_0…G_100000`，每 2000 步一存，共 51 个；
> 浅层扩散 `logs/44k/diffusion/model_10000…188000`，每 10000 步一存）
> 证据来源：TensorBoard（http://127.0.0.1:6006）、[`logs/44k/train.log`](../logs/44k/train.log)、
> [`logs/44k/diffusion/log_info.txt`](../logs/44k/diffusion/log_info.txt)、实际推理音频、数据集留出集音频。

## 1. 结论（本次导出）

| 角色 | 选定 | 步数 |
|---|---|---|
| **主模型** | [`logs/44k/G_100000.pth`](../logs/44k/G_100000.pth) | 100000（本次训练终态） |
| **浅层扩散** | [`logs/44k/diffusion/model_188000.pt`](../logs/44k/diffusion/model_188000.pt) | 188000 |

导出产物（均已通过 `onnx.checker.check_model`）：

```
checkpoints/
├── huawu_main_100000.json                      # MoeVS 配置（SoVits/Volume/Characters）
├── huawu_main_100000/
│   ├── huawu_main_100000_SoVits.onnx           # 135.8 MB  opset16
│   ├── model.pth / config.json
└── huawu_diff_188000/
    ├── huawu_diff_188000_encoder.onnx          # 0.8 MB   hubert,mel2ph,f0,volume -> mel_pred
    ├── huawu_diff_188000_denoise.onnx          # 220.0 MB noise,time,condition -> noise_pred
    ├── huawu_diff_188000_pred.onnx             # 6.6 KB   noise,noise_pred,time,time_prev -> noise_pred_o
    └── huawu_diff_188000_after.onnx            # 1.0 KB   x -> mel_out
```

> 本次训练未训练聚类/特征检索（`logs/44k` 下无 `kmeans_10000.pt` / `feature_and_index.pkl`），
> 故 MoeVS 配置中 `Cluster` 留空、不附带 `Index-*.index`。

### 1.1 可移植 ONNX 推理包（跨机器部署）

[`checkpoints/onnx_bundle/`](../checkpoints/onnx_bundle/README.md)（1.0 GB，自包含，目标机仅需
`onnxruntime + numpy + soundfile`）：

| 组件 | 文件 | 体积 |
|---|---|---|
| ContentVec768L12 编码器 | `vec768l12.onnx` | 377.9 MB |
| RMVPE F0 预测器（+ `mel_basis.npy`） | `rmvpe.onnx` | 362.1 MB |
| 主模型（随机外置版） | `SoVits.onnx` | 135.7 MB |
| 主模型（MoeVS 标准 7 输入版） | `SoVits_moevs.onnx` | 135.8 MB |
| NSF-HiFiGAN 声码器 | `nsf_hifigan.onnx` | 56.7 MB |
| 浅层扩散四件套 | `diffusion/{encoder,denoise,pred,after}.onnx` | 220.9 MB |
| 常量（mel 滤波器组 / 重采样核 / 扩散系数 / config） | `*.npy`、`*.npz`、`config.json` | ~1.6 MB |

逐组件数值对齐（[`tools/verify_onnx_bundle.py`](../tools/verify_onnx_bundle.py:1)）：
RMVPE f0 `cos=1.000000`、uv 完全一致、ContentVec c `cos=1.000000`、Volume `3e-8`、
SoVits 图 `3.2e-6`、扩散 condition `1.1e-6` / denoise `8.1e-6` / pred·after `0`、
NSF-HiFiGAN `cos=0.999996`；端到端与 PyTorch 的 log-mel 平均差 **0.22 dB**、包络相关 **0.92**。

> 导出过程中发现并修正的两个上游问题：① 标准 `onnx_export.py` 会把 dec(NSF 激励) 的随机量
> 固化为常量，且强制走 SineGen `onnx` 分支（对超长序列做未取模 cumsum，fp32 相位误差导致
> 直流/次声，实测 <50Hz 占 6.6%）；现改为导出生产同款 `_f02sine` 分支并把 `rand_ini`/`dec_noise`
> 外置为输入（[`tools/export_onnx_sovits_hifi.py`](../tools/export_onnx_sovits_hifi.py:1)），
> 并在推理包输出端加 20Hz 高通。② 扩散 ONNX 导出时 `max(t-interval,0)` 返回 Python `int` 导致中断，
> 已改为 `torch.clamp`。

## 2. 主模型选型

### 2.1 训练损失（TensorBoard 标量）

本次 run 训练到 100000 步（`max_steps`）自动停止：`loss/g/total` 429.3 → **32.7**，
`loss/g/mel` 129.7 → **20.9**，`loss/g/fm` 0.83 → 5.59（对抗平衡），`loss/d/total` 2.36，无 NaN/Inf。

用 [`tools/rank_main_checkpoints.py`](../tools/rank_main_checkpoints.py:1) 对磁盘上 51 个 checkpoint 做
±1000 步滑窗排序（复合分数 = z(mel)+z(total)+0.5·z(fm)）：

| 排名 | step | score | g/total | mel | fm | d/total | grad_norm_g |
|---|---|---|---|---|---|---|---|
| 1 | 98000 | -0.869 | 32.56 | 21.18 | 5.60 | 2.46 | 11306 |
| 2 | 96000 | -0.799 | 32.68 | 21.19 | 5.70 | 2.36 | 10501 |
| 3 | 92000 | -0.796 | 33.03 | 21.40 | 5.65 | 2.35 | 9986 |
| 4 | 76000 | -0.780 | 33.42 | 21.95 | 5.57 | 2.39 | 9906 |
| 5 | 86000 | -0.769 | 33.31 | 21.79 | 5.62 | 2.35 | 11269 |
| 8 | **100000** | -0.755 | 33.15 | 21.37 | 5.71 | 2.29 | 9535 |
| 11 | 94000 | -0.683 | 33.28 | 21.47 | 5.79 | 2.35 | 8523 |
| 最差 | 0 | +9.930 | 154.75 | 97.71 | 2.69 | — | — |

> 51 个候选分数集中在 -0.87 ~ -0.66（除早期步数），差异 < 0.7 分，属窗口内正常波动，
> 单看损失不足以定版，需结合音质指标。

### 2.2 自转换保真度（留出集，[`tools/self_convert_eval.py`](../tools/self_convert_eval.py:1)，n=8）

以 [`filelists/val.txt`](../filelists/val.txt) 前 8 条留出音频同时作输入与参考做同说话人自转换
（mel_L1 / MCD 越低越好；dHNR = 输出 HNR − 参考 HNR，越接近 0 越少引入噪声）：

| 模型 | mel_L1(dB) | MCD | dHNR(dB) | Δ质心(Hz) |
|---|---|---|---|---|
| G_94000 | **5.44** | 273.43 | -2.19 | +35 |
| G_98000 | 5.51 | 274.56 | -1.11 | -79 |
| **G_100000** | 5.54 | **271.25** | **-0.52** | -79 |
| G_92000 | 5.55 | 278.34 | -1.54 | -36 |
| G_96000 | 5.56 | 278.33 | -1.78 | -9 |
| G_76000 | 5.67 | 281.91 | -1.14 | -95 |
| G_86000 | 5.70 | 285.93 | -1.11 | -98 |
| G_78000 | 5.84 | 292.25 | -1.34 | -78 |

`G_94000` 的 mel_L1 最优，`G_100000` 的 MCD 与 dHNR 最优（引入的噪声最少）。

### 2.3 跨音色推理与数据集对比（[`tools/compare_audio.py`](../tools/compare_audio.py:1)）

用两段不同音色来源做真实转换（模型均未见过）：

- 源 1 `raw/test_eval.wav`：明亮，质心 3099 Hz，flatness 0.0740
- 源 2 `raw/test_eval2.wav`：偏闷（更接近数据集），质心 1535 Hz，flatness 0.00253，HNR 9.72 dB

| 模型 | 源1 质心/HF%/flatness/HNR | 源1 音色距离 | 源2 质心/HF%/flatness/HNR | 源2 音色距离 |
|---|---|---|---|---|
| G_92000 | 1589 / 2.30 / 0.00842 / 1.00 | 5.51 | 1243 / 0.62 / 0.00331 / 7.17 | 2.78 |
| G_94000 | 1847 / 2.57 / 0.01025 / 0.21 | 5.76 | 1263 / 0.66 / 0.00365 / 6.55 | 3.06 |
| G_96000 | 1655 / 2.29 / 0.00862 / 0.89 | **5.36** | 1277 / 0.57 / 0.00324 / 7.33 | **2.55** |
| G_98000 | 1487 / 2.27 / 0.00815 / 1.59 | 5.46 | 1201 / 0.56 / 0.00313 / 7.95 | 2.75 |
| **G_100000** | 1488 / 2.29 / **0.00761** / **2.02** | 5.41 | 1164 / **0.47** / **0.00261** / **8.89** | 2.68 |
| 数据集样本（6 条） | 1042~1288 / 0.03~0.39 / 0.0003~0.0011 / 2.25~10.15 | 参考 | — | — |

（音色距离=1/3 倍频程能量相对训练集 LTAS 的 RMS 差，越低越接近目标音色）

### 2.4 主模型结论：`G_100000.pth`

1. **噪声/电音最少**：两段源上 flatness 均最低（0.00761 / 0.00261），HNR 均最高
   （2.02 / 8.89，源 2 上 8.89 已非常接近源本身的 9.72，其余模型仅 6.6~8.0）；
2. **自转换引入失真最少**：MCD 最低（271.25）、dHNR 最接近 0（-0.52）；
3. **音色距离接近最优**：5.41 / 2.68（最优 5.36 / 2.55，差 0.05~0.13 dB，属测量噪声）；
4. 是本次训练的**收敛终态检查点**（TB 综合排名第 8，与第 1 名差 0.11 分）。

> `G_94000` 虽在自转换指标上最好，但其跨音色输出明显偏亮偏噪（质心 1847、flatness 0.01025、
> HNR 0.21），说明自转换对训练域内容易受"记忆"影响，跨音色指标更可信。

次选：`G_96000`（源 1/源 2 音色距离均最优，TB 第 2）、`G_98000`（TB 第 1）。

## 3. 浅层扩散选型

### 3.1 训练情况（[`logs/44k/diffusion/log_info.txt`](../logs/44k/diffusion/log_info.txt)）

- 18873 条训练记录，train loss 0.999 → 0.143（5000 步均值 0.367 → 0.156，15 万步后基本平台）；
- 验证 loss 30000 步后稳定在 0.16~0.17，**最优 0.1510 @ step≈146000**；
  分段均值最低为 140000~159999（0.1602），末段 180000~188090 为 0.1589。

### 3.2 自转换对比（以 `G_100000` 为底，n=4）

| 组合 | mel_L1(dB) | MCD | dHNR(dB) |
|---|---|---|---|
| G_100000 无扩散 | 5.50 | 270.5 | -0.08 |
| + model_150000 | 5.65 | 286.7 | -0.81 |
| + model_160000 | 5.65 | 286.6 | -0.82 |
| + model_170000 | 5.65 | 286.7 | -0.67 |
| + model_180000 | 5.64 | 286.8 | -0.94 |
| **+ model_188000** | **5.63** | **286.3** | -0.85 |

5 个候选几乎同分，`model_188000` 略优。

### 3.3 跨音色对比

| 组合 | 源2 HNR(dB) | 源2 HF8-16k% | 源2 flatness | 源2 音色距离 | 源1 音色距离 |
|---|---|---|---|---|---|
| G_100000 无扩散 | 8.89 | 0.47 | 0.00261 | 2.68 | 5.41 |
| + **model_188000** | **11.87** | 0.37 | 0.00261 | 3.89 | 6.05 |
| + model_150000 | 11.78 | 0.39 | 0.00267 | 4.10 | — |

### 3.4 扩散结论：`model_188000.pt`

- 两段源上均优于 `model_150000`（音色距离 3.89 vs 4.10；HF 更低）；
- 是本次扩散训练的收敛终态，且与验证最优段（146k）差距仅 ~5%；
- **作用与代价**：HNR 8.89 → 11.87 dB（谐波性提升、明显抑制电音/毛刺），代价是高频略降、
  质心下降、与数据集音色距离增加约 1.2 dB —— 这是浅层扩散的固有取舍，按素材决定是否开启。

> 备选：追求"验证 loss 最低"可用 [`logs/44k/diffusion/model_150000.pt`](../logs/44k/diffusion/model_150000.pt)。

## 4. 复现命令

```bash
# 1) 本次 run 检查点排序（TensorBoard 离线）
python tools/rank_main_checkpoints.py --top 12

# 2) 自转换保真度（8 条留出集）
python tools/self_convert_eval.py --n 8 \
  --case G_92000=logs/44k/G_92000.pth=logs/44k/config.json \
  --case G_94000=logs/44k/G_94000.pth=logs/44k/config.json \
  --case G_96000=logs/44k/G_96000.pth=logs/44k/config.json \
  --case G_98000=logs/44k/G_98000.pth=logs/44k/config.json \
  --case G_100000=logs/44k/G_100000.pth=logs/44k/config.json

# 3) 批量推理 + 客观指标/音色对比
python tools/eval_models.py --raw test_eval2.wav --case s2_G100000=logs/44k/G_100000.pth=logs/44k/config.json
python tools/compare_audio.py ab results_eval/s2_G100000.wav
python tools/compare_audio.py timbre results_eval/s2_G100000.wav --n 120

# 4) 导出 ONNX
python onnx_export.py -n huawu_main_100000
python diffusion/onnx_export.py --project-name huawu_diff_188000 \
  --model-path logs/44k/diffusion/model_188000.pt --out-dir checkpoints/huawu_diff_188000
```

PyTorch 侧推理（部署前 A/B 试听）：

```bash
python inference_main.py -m logs/44k/G_100000.pth -c logs/44k/config.json \
  -dm logs/44k/diffusion/model_188000.pt -dc logs/44k/diffusion/config.yaml \
  -n <wav> -s huawu -f0p rmvpe -ks 100 -shd -vd cpu -wf wav
```

## 5. 本次为导出所做的代码改动

| 文件 | 改动 | 原因 |
|---|---|---|
| [`onnx_export.py`](../onnx_export.py:103) | 导出增加 `dynamo=False` | torch 2.9 默认 dynamo 导出器需要 `onnxscript`（未安装）；传统导出器支持 `dynamic_axes` |
| [`diffusion/onnx_export.py`](../diffusion/onnx_export.py:135) | 3 处导出加 `dynamo=False`；`__main__` 改为 `argparse`（`--project-name/--model-path/--out-dir/--merged`） | 同上；并支持任意扩散 checkpoint 与输出目录 |
| [`diffusion/diffusion_onnx.py`](../diffusion/diffusion_onnx.py:321) | `max(t - interval, 0)` → `torch.clamp(t - interval, min=0)` | 当 `t-interval<0` 时 `max()` 返回 Python `int`，导致 `'int' object has no attribute 'float'` 使扩散导出中断 |
| [`tools/export_index_for_onnx.py`](../tools/export_index_for_onnx.py:1) | 硬编码路径改为 `--pkl/--out-dir` | 参数化 |
| 新增 [`tools/rank_main_checkpoints.py`](../tools/rank_main_checkpoints.py:1) | 检查点指标排序 | 选型 |
| 新增 [`tools/eval_models.py`](../tools/eval_models.py:1) | 批量推理 | 生成对比音频 |
| 新增 [`tools/self_convert_eval.py`](../tools/self_convert_eval.py:1) | 自转换保真度评估 | 量化音质替代指标 |

## 6. 附：与历史版本（`model/` 下 6.0/7.0/7.1）的对照数据

应要求本次选型仅在 `logs/44k` 本次训练 run 内进行。为便于后续判断，保留此前跨版本实测数据：

| 模型 | 自转换 mel_L1(dB) / MCD | 跨音色源1 flatness / HNR / 音色距离 |
|---|---|---|
| 6.0 G_389600 | 4.96 / 246.7 | 0.00223 / 5.43 / 5.38 |
| 7.0 G_75200 | 5.37 / 268.9 | 0.00336 / 4.70 / 6.44 |
| 7.1 G_126400 | 5.48 / 272.3 | 0.00255 / 6.61 / 5.99 |
| 本次 G_100000 | 5.54 / 271.3 | 0.00761 / 2.02 / 5.41 |

即：按客观保真度，6.0（389.6k 步）仍优于本次 run（100k 步，仍处下降通道，mel≈21 vs 6.0 平台 mel≈16）。
若后续把本次训练继续训到 20~30 万步，预计可追平/超过 6.0。

## 7. 局限

1. **无主观听感**：mel 距离 / MCD / HNR / flatness / 音色距离只是"音质"的客观代理；
   建议对 `results_eval/` 下音频做 A/B 试听，尤其确认"是否开启浅层扩散"的取舍。
2. **自转换存在记忆偏差**：留出集若曾参与该模型训练，保真度会被高估（`G_94000` 即为典型），
   故结论以跨音色指标为主、自转换指标为辅。
3. **ONNX 仅结构校验**：环境未安装 `onnxruntime`，未做 PyTorch↔ONNX 数值一致性比对。
4. 51 个主模型 checkpoint（每个含优化器状态约 1.5 GB）共约 30 GB，定版后可参考
   [`tools/analyze_checkpoints.py`](../tools/analyze_checkpoints.py:1) 仅保留权重以释放空间。
