<!-- 中文 -->

# SoftVC VITS Singing Voice Conversion For Intel XPU

![wmm](./doc/img/1701608234384.png)

📻 基于So-VITS-SVC模型的音色训练推理框架，专为Intel显卡优化设计。

## ⚠️ 重要声明

1. 此项目**仅支持Intel独显/核显(XPU)**，不支持NVDIA、AMD等GPU，请勿在其他平台使用；
2. 本项目为开源、离线的项目，**不能收集任何用户信息或获取用户输入数据**，不负责任何用户输入。本项目**不向任何组织、个人提供任何形式的支持**，故一切基于本项目训练的 AI 模型和合成的音频都**与本项目贡献者无关**。一切由此造成的问题**由使用者自行承担**；
3. 本项目只是一个框架项目，没有任何模型，任何二次分发的项目都与这个项目的贡献者无关；
4. 请自行解决数据集授权问题，**禁止使用非授权数据集进行训练**。任何由于使用非授权数据集进行训练造成的问题，需**自行承担全部责任和后果**。

## 📗 项目简介

本项目是基于[So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc)项目，原项目版本为`4.1-Stable`，使用PyTorch+XPU，专为Intel显卡优化。用于声音音色转换、AI翻唱等功能。通过SoftVC内容编码器提取源音频语音特征。

## 🚗 已经测试过的GPU硬件

+ Intel Iris Xe Graphics eligible
+ Intel Arc A380 Graphics Card
+ Intel Arc A770 Graphics Card
  

## 🧪 环境配置

### 1. 安装Python环境

本项目使用Python 3.11，理论上支持更高版本的Python，但尚未进行充分测试。  
由于PyTorch+XPU最低要求Python 3.10，因此需要安装Python 3.10及以上版本。
```
# Windows
winget install --id Python.Python.3.11

# Ubuntu
sudo apt install python3.11 python3.11-dev python3.11-venv

# Fedora
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. 创建虚拟环境

在项目根目录下执行终端命令创建虚拟环境
```bash
# Windows
py -3.11 -m venv venv

# Linux
python3.11 -m venv venv
```

激活虚拟环境
```bash
# Windows
venv\Scripts\Activate.ps1

# Linux
source venv/bin/activate
```

### 3. 安装项目依赖

**安装PyTorch**

当前PyTorch已官方支持Intel显卡，因此只需安装PyTorch即可，无需再安装IPEX。
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 下载慢或频繁终端可以使用镜像源下载
# 南京大学
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/xpu
# 上海交通大学
pip install torch torchvision torchaudio --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/xpu
```

**安装其余依赖**

```bash
pip install -r requirements.txt
```

## 🔨 预先下载的模型文件

### 编码器

以下编码器需要选择一个使用
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### 1. 若使用contentvec作为声音编码器（推荐）

vec768l12与vec256l9需要该编码器，请从以下两个链接中下载（**二选一**）。

+ [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
+ [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)

将文件名重命名为`checkpoint_best_legacy_500.pt`后，放在`pretrain`目录下。

#### 2. 若使用hubertsoft作为声音编码器

下载模型[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)，放在`pretrain`目录下。

#### 3. 若使用Whisper-ppg作为声音编码器

+ 下载模型[medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)，该模型适配`whisper-ppg`。
+ 下载模型[large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)，该模型适配`whisper-ppg-large`。

放在`pretrain`目录下。

#### 4. 若使用cnhubertlarge作为声音编码器

下载模型[chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt).  
放在`pretrain`目录下。

#### 5. 若使用dphubert作为声音编码器

下载模型[DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth).  
放在`pretrain`目录下。

#### 6. 若使用WavLM作为声音编码器

下载模型[WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), 该模型适配`wavlmbase+`。  
放在`pretrain`目录下。

#### 7. 若使用OnnxHubert/ContentVec作为声音编码器

下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main).  
放在`pretrain`目录下。

### 预训练底模（可选）

使用预训练底模可以获得更快的训练速度和更好的训练效果。

+ 预训练底模文件： `G_0.pth` `D_0.pth`
  + 放在`logs/44k`目录下

+ 扩散模型预训练底模文件： `model_0.pt`
  + 放在`logs/44k/diffusion`目录下
  
SoVits底模文件：[ms903/sovits4.0-768vec-layer12 at main](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/tree/main/sovits_768l12_pre_large_320k)  

扩散模型引用了 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的 Diffusion Model，底模与 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的扩散模型底模通用，可以去 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 获取扩散模型的底模。

### NSF-HIFIGAN（可选）

如果使用**NSF-HIFIGAN 增强器**或**浅层扩散**的话，需要下载预训练的 NSF-HIFIGAN 模型。  
预训练的 NSF-HIFIGAN 声码器：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip).  
解压后，将四个文件放在`pretrain/nsf_hifigan`目录下。

### RMVPE（可选）

如果使用rmvpeF0预测器的话，需要下载预训练的RMVPE模型。  
下载模型[rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，解压缩`rmvpe.zip`，并将其中的`model.pt`文件改名为`rmvpe.pt`并放在`pretrain`目录下。

## 📚 准备训练数据

准备几段仅单人人声、无背景音乐的音频作为训练数据，并保存为wav文件。  
建议训练音频包含唱歌和普通讲话音频。  
可以使用[UVR5](https://github.com/Anjok07/ultimatevocalremovergui/)进行人声提取工作。

### 1. 音频切片

将训练音频进行切片，可以使用[audio-slicer-GUI](https://github.com/flutydeer/audio-slicer)。

切片好的音频按照下列文件结构将数据集放入`dataset_raw`目录。
```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
对于每一个音频文件的名称并没有格式的限制，但为了后续方便，建议全部采用英文命名。  
不过文件格式必须为wav。  
以及，可以自定义说话人名称。
```
dataset_raw
└───HuaWuNyako
    ├───1.wav
    ├───a.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

### 2. 重采样至44100Hz单声道

```
python resample.py
```

注意：虽然本项目拥有重采样、转换单声道与响度匹配的脚本 resample.py，但是默认的响度匹配是匹配到 0db。这可能会造成音质的受损。而 python 的响度匹配包 pyloudnorm 无法对电平进行压限，这会导致爆音。所以建议可以考虑使用专业声音处理软件如adobe audition等软件做响度匹配处理。  
若已经使用其他软件做响度匹配，可以在运行上述命令时添加--skip_loudnorm跳过响度匹配步骤。
```bash
python resample.py --skip_loudnorm
```

### 3. 自动划分训练集、验证集，以及自动生成配置文件

```bash
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder参数可选:

+ vec768l12（默认）
+ vec256l9
+ hubertsoft
+ whisper-ppg
+ whisper-ppg-large
+ cnhubertlarge
+ dphubert
+ wavlmbase+

若使用响度嵌入，需要增加--vol_aug参数。
```bash
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

#### 配置文件

此时可以在生成的`config.json`与`diffusion.yaml`修改部分参数

**config.json**

+ keep_ckpts: 训练时保留最后几个模型，0为保留所有，默认只保留最后3个
+ all_in_mem: 加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 远大于 数据集体积时可以启用
+ batch_size: 单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
+ c_mel: Mel谱损失权重，默认值45，控制频谱域重建误差的重要性
+ c_kl: KL散度损失权重，默认值1.0，控制变分自编码器的正则化强度  
+ c_fm: 特征匹配损失权重，默认值0.5，控制生成器在对抗训练中特征匹配损失的影响程度。该参数帮助生成器学习更接近真实数据的中间特征表示，建议值0.1-1.0，过高可能导致训练不稳定
+ vocoder_name: 选择一种声码器，默认为nsf-hifigan。
  + 声码器列表
    + nsf-hifigan
    + nsf-snake-hifigan


**diffusion.yaml**

+ cache_all_data: 加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量远大于数据集体积时可以启用
+ duration: 训练时音频切片时长，可根据显存大小调整，注意，该值必须小于训练集内音频的最短时间！
+ batch_size: 单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
+ timesteps: 扩散模型总步数，默认为 1000。
+ k_step_max: 训练时可仅训练k_step_max步扩散以节约训练时间，注意，该值必须小于timesteps，0 为训练整个扩散模型，注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！

### 4. 生成 hubert 与 f0

```bash
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor可选参数
+ crepe
+ dio
+ pm
+ harvest
+ rmvpe
+ fcpe

如果训练集过于嘈杂，建议使用 crepe 处理 f0。  
如果省略 f0_predictor 参数，默认值为 rmvpe。

尚若需要浅扩散功能，需要增加--use_diff 参数。
```bash
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

#### 💻 使用 CPU 进行预处理

如果您想使用 CPU 而不是 XPU/GPU，可以通过 `--device` 参数指定：

```bash
# 强制使用 CPU 模式
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu
```

**CPU 模式的优势**:
- ✅ 完全避免 XPU 多进程问题
- ✅ 可以使用更多进程（推荐 4-8 个）
- ✅ 更稳定，不会导致系统死机
- ✅ 适合没有独立显卡或显存不足的情况

**CPU 模式推荐配置**:

```bash
# 4核CPU：使用2-3个进程
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 3

# 6核CPU：使用4-5个进程（推荐）
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 4

# 8核CPU：使用6个进程
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 6

# 16核及以上：最多使用12-16个进程（需要大内存）
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 12
```

**自动优化**: 代码会自动检测您的 CPU 核心数和内存大小，计算最优进程数并给出建议。

**用户控制权**: 程序不会强制限制进程数，您可以自由设置任何值。但如果超过系统资源承受能力，会显示详细警告。

**内存占用参考**:
- 不使用 `--use_diff`：每进程约 2-2.5GB
- 使用 `--use_diff`：每进程约 3-4GB

**建议**: CPU 进程数设置为物理核心数的 75%，以平衡速度和系统响应性。

#### ⚠️ 重要：Intel Arc A770 多进程使用警告

**已知问题**: 根据 Intel 官方 OpenVINO 2025.2 发布说明，在 Intel Arc A770 上同时运行多个进程可能导致系统死机。

**技术原因**:
- Intel Arc A770 的驱动程序在多进程并发访问时存在稳定性问题
- 每个进程加载大型模型会占用大量内存和显存资源
- PyTorch 线程与 multiprocessing 的资源竞争可能引发死锁

**推荐配置方案**:

##### 方案 1：最大稳定性（强烈推荐）
```bash
# 单进程模式，最稳定，适合大多数用户
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 1
```
- ✅ 完全避免多进程冲突
- ✅ 内存占用最低
- ✅ 不会导致系统死机
- ⚠️ 速度相对较慢，但最安全

##### 方案 2：适度加速（Arc A770 用户）
```bash
# 2-4 个进程，平衡速度和稳定性
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 2
```
- ✅ 比单进程快 1.5-2 倍
- ⚠️ 需要监控系统温度和内存
- ⚠️ 不建议超过 4 个进程
- ❌ 仍有小概率导致系统不稳定

##### 方案 3：CPU 多核加速（非 XPU 用户）
```bash
# CPU 用户可以适当增加进程数，但仍需限制
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 4
```
- ✅ 充分利用 CPU 多核
- ⚠️ 建议设置为 CPU 核心数的一半
- ⚠️ 最多不超过 8 个进程

**自动保护机制**:
- ✅ 代码会自动检测 Arc A770 并给出警告
- ✅ 自动限制 Arc A770 最大进程数为 4
- ✅ 每个进程的 PyTorch 线程数限制为 1
- ✅ 设置 OMP_NUM_THREADS 和 MKL_NUM_THREADS 环境变量
- ✅ 详细的日志输出，方便监控处理进度

**监控建议**:
1. 打开任务管理器，监控 CPU 和内存使用率
2. 如果内存使用超过 80%，立即停止并减少进程数
3. 监控系统温度，确保散热良好
4. 首次使用时建议从 `--num_processes=1` 开始测试

执行完以上步骤后，`dataset`目录便是预处理完成的数据，此时`dataset_raw`文件夹可以删除。

### 5. 开始训练

```
python train.py -c configs/config.json -m 44k
```

训练过程中，TensorBoard日志将写入`logs/44k`目录，可以通过以下命令查看：

```
tensorboard --logdir logs/44k
```

如果需要中断训练，可以按`Ctrl+C`终止训练进程。

训练检查点将保存在`logs/44k`目录下，包括：
- `G_*.pth`：生成器模型检查点
- `D_*.pth`：判别器模型检查点

浅扩散模型训练
```
# 注意：扩散模型需要先用 --use_diff 预处理数据：
# python preprocess_hubert_f0.py --f0_predictor dio --use_diff

# 然后独立训练扩散模型（不影响 SoVITS 训练）
python train_diff.py -c configs/diffusion.yaml
```

如果出现训练不稳定，经常中断的情况，可以使用`supervisor`进行进程守护训练，防止模型训练中断。  
安装supervisor
```
# Ubuntu
sudo apt install supervisor

# Fedora
sudo dnf install supervisor

# Windows
pip install supervisor-win
```

模型训练
```
# 主模型训练
supervisord -n -c train_supervisord.conf

# 浅扩散模型训练
supervisord -n -c train_supervisord_diff.conf
```

可以使用TensorBoard监控训练状态
```
tensorboard --logdir logs/44k --bind_all
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下

## 📝 模型推理

### 1. 实时变声（API服务）

启动实时变声 API 服务（配合 VST 插件使用，默认监听 `0.0.0.0:6842`）：
```bash
python flask_api.py
```
> 注意：本项目不提供 `gui.py` 图形界面。使用前请先修改 [`flask_api.py`](flask_api.py) 中的 `model_name`、`config_name`、`cluster_model_path` 为您的模型路径。

### 2. 批量推理

使用命令行进行批量推理：
```bash
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi"
```

**更多常用推理命令示例**：

```bash
# 1. 变调推理：整体升高 2 个半音（-t 2），使用 rmvpe F0 预测器（-f0p rmvpe，歌曲更稳）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 2 -s "buyizi" -f0p rmvpe

# 2. 语音转换：开启自动音高预测（-a，仅适合说话，唱歌会严重跑调）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "讲话.wav" -t 0 -s "buyizi" -a

# 3. 聚类音色控制：使用聚类模型，占比 0.5（-cr 0.5，需先训练 cluster/train_cluster.py）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -cm "logs/44k/kmeans_10000.pt" -cr 0.5

# 4. 特征检索：使用特征检索索引，占比 0.5（-fr，需先训练 train_index.py）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -fr -cr 0.5

# 5. 浅扩散：解决电音问题（-shd，需先训练扩散模型，必须指定 -dm/-dc 扩散模型路径与配置）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -dm "logs/44k/diffusion/model_0.pt" -dc "logs/44k/diffusion/config.yaml" -shd -ks 100 -vd cpu

# 6. 纯扩散：仅使用扩散模型推理（-od，需完整训练的扩散模型，必须指定 -dm/-dc）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -dm "logs/44k/diffusion/model_0.pt" -dc "logs/44k/diffusion/config.yaml" -od -ks 100

# 7. 增强器：开启 NSF-HIFIGAN 增强器（-eh，训练数据少时可改善音质）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -eh

# 8. 角色融合：开启说话人混合（-usm，需在 spkmix.py 中配置）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -usm

# 9. 指定设备：模型与声码器均使用 CPU 推理（-d cpu -vd cpu）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -d cpu -vd cpu

# 10. 批量推理：一次转换多个音频（-n 后可跟多个文件名，-t 逐一对应）
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "song1.wav" "song2.wav" -t 0 0 -s "buyizi"
```

必需参数：
- `-m` | `--model_path`：模型路径
- `-c` | `--config_path`：配置文件路径
- `-n` | `--clean_names`：输入音频文件名，放在raw目录下
- `-t` | `--trans`：音高调整（半音单位，支持正负值）
- `-s` | `--spk_list`：目标说话人名称
- `-cl` | `--clip`：音频强制切片时长（秒），默认0为自动切片

可选参数：
- `-lg` | `--linear_gradient`：音频切片间的交叉淡化时长（秒），强制切片后出现破音时调整（默认：0）
- `-f0p` | `--f0_predictor`：F0预测器选择（选项：crepe, pm, dio, harvest, rmvpe, fcpe；默认：pm）。注意：crepe使用均值滤波处理原始F0
- `-a` | `--auto_predict_f0`：启用自动音高预测，适合语音转换（歌声转换时不建议开启，可能导致严重跑调）
- `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径。留空则使用各方案默认路径
- `-cr` | `--cluster_infer_ratio`：聚类或特征检索占比（范围：0-1）。如果没有训练聚类模型或特征检索则设为0
- `-eh` | `--enhance`：启用NSF_HIFIGAN增强器。此选项可能改善训练数据较少的模型音质，但对于训练充分的模型可能降低音质（默认：禁用）
- `-shd` | `--shallow_diffusion`：启用浅层扩散以解决电音问题（默认：禁用）。注意：启用此选项时NSF_HIFIGAN增强器将被禁用
- `-usm` | `--use_spk_mix`：启用角色融合/动态声音混合
- `-lea` | `--loudness_envelope_adjustment`：输入源与输出的响度包络混合比例。数值越接近1使用越多的输出响度包络
- `-fr` | `--feature_retrieval`：启用特征检索（禁用聚类模型）。启用时cm和cr参数分别变为特征检索索引路径和混合比例
- `-d` | `--device`：主推理设备。不指定时自动选择（Intel环境默认`xpu`）
- `-vd` | `--vocoder_device`：声码器（NSF-HiFiGAN）合成设备，可指定`cpu`或`xpu`。默认不指定时跟随主推理设备

浅层扩散设置：
+ `-dm` | `--diffusion_model_path`：扩散模型路径
+ `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
+ `-ks` | `--k_step`：扩散步数。数值越高结果越接近扩散模型输出（默认：100）
+ `-od` | `--only_diffusion`：纯扩散模式。此模式不会加载SoVITS模型，仅使用扩散模型进行推理
+ `-se` | `--second_encoding`：二次编码。在浅层扩散前对原始音频进行额外编码。这是实验性选项，效果不定

> **⚠️ XPU 声码器高频噪声问题（重要）**
> 在 Intel XPU 上，`nsf_hifigan` 声码器的 `noise_convs`（Conv1d）内核在处理低幅度谐波信号（`har_source`）时存在数值错误，会导致**16kHz 以上频段出现白噪声**（实测输出 8k+ 高频占比高达 30%+，且整体响度偏弱）。
> - **现象**：启用浅层扩散（`-shd`）时，输出音频出现明显高频嘶声/白噪；不启用扩散（纯 SoVITS）则正常。
> - **根因**：XPU 后端 Conv1d 内核缺陷（PyTorch XPU 后端问题，非本项目代码问题；禁用 oneDNN、输入缩放均无法规避）。
> - **解决办法**：推理时指定声码器在 CPU 合成，扩散模型与主模型仍留在 XPU，音质与速度兼顾：
>   ```bash
>   python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -dm "logs/44k/diffusion/model_0.pt" -dc "logs/44k/diffusion/config.yaml" -shd -ks 100 -vd cpu
>   ```
>   加上 `-vd cpu` 后，输出 8k+ 高频占比从 30%+ 降至 <1%，音质恢复干净。

注意：使用whisper-ppg语音编码器进行推理时，需设置`--clip`为25，`--lg`为1。否则无法正常推理。

### f0预测器比较

以下是各个f0预测器算法在推理时的优缺点：
| 预测器  |              优点              |                     缺点                     |
| :-----: | :----------------------------: | :------------------------------------------: |
|   pm    |         速度快，占用低         |                 容易出现哑音                 |
|  crepe  |        基本不会出现哑音        | 显存占用高，自带均值滤波，因此可能会出现跑调 |
|   dio   |               -                |                   可能跑调                   |
| harvest |       低音部分有更好表现       |           其他音域就不如别的算法了           |
|  rmvpe  | 六边形战士，目前最完美的预测器 |     几乎没有缺点（极端长低音可能会出错）     |


### 自动f0预测（可选）

4.0模型训练过程会训练一个f0预测器。对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动预测，但转换歌声时请不要启用此功能，会**严重跑调**。  
在`inference_main`中设置`auto_predict_f0`为`true`即可。

### 聚类音色泄漏控制（可选）

聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点。  
使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低。

训练：
```bash
python cluster/train_cluster.py
```
> 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`  
> 聚类模型目前可以使用 gpu 进行训练，执行`python cluster/train_cluster.py --gpu`

推理：
> 在`inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/kmeans_10000.pt`  
> 在`inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比。

训练：

首先需要在生成 hubert 与 f0 后执行：
```
python train_index.py -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

推理：
> 需要首先指定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案  
> 在`inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/feature_and_index.pkl`  
> 在`inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可

## 📦 模型压缩

移除模型中的训练信息以减小文件大小（约为原始大小的1/3）：

```
python compress_model.py -c="configs/config.json" -i="logs/44k/G_<模型名称>.pth" -o="logs/44k/release.pth"
```

## 📤 ONNX导出

将模型导出为ONNX格式以便部署：

1. 在项目根目录新建 `checkpoints/<模型文件夹名>` 文件夹；
2. 将训练好的模型重命名为 `model.pth`、配置文件重命名为 `config.json`，放入该文件夹；
3. 执行导出命令（`-n` 指定模型文件夹名）：

```bash
python onnx_export.py -n <模型文件夹名>
```

导出完成后，会在 `checkpoints/<模型文件夹名>/` 下生成 `<模型文件夹名>_SoVits.onnx`，同时在 `checkpoints/` 下生成对应的 MoeVS 配置 `<模型文件夹名>.json`。

> 注意：本项目使用 [`onnx_export.py`](onnx_export.py) 导出（支持说话人混合），不是 `export_onnx.py`。

### 完整导出（编码器 / 声码器 / F0，用于跨机器部署）

`onnx_export.py` 只导出 SoVits 主体。若要在**没有 PyTorch 环境**的机器上推理，还需导出内容编码器、
声码器与 F0 预测器。以下脚本会把它们一并导出，并汇总为自包含推理包：

```bash
# 1) 主模型（MoeVoiceStudio 用，7 输入）—— 需先把权重/配置放入 checkpoints/<模型文件夹名>/
python onnx_export.py -n <模型文件夹名>

# 2) 内容编码器 ContentVec768L12（用 fairseq 原始模型子链路导出，与推理数值完全一致）
python tools/export_onnx_encoder.py --out checkpoints/onnx_bundle/vec768l12.onnx

# 3) F0 预测器 RMVPE（附带 log-mel 滤波器组常量）
python tools/export_onnx_rmvpe.py --out checkpoints/onnx_bundle/rmvpe.onnx

# 4) 声码器 NSF-HiFiGAN（激励随机量外置，保证可复现）
python tools/export_onnx_vocoder.py --out checkpoints/onnx_bundle/nsf_hifigan.onnx

# 5) SoVits 随机外置版（修复导出图随机被固化、以及直流/次声问题）
python tools/export_onnx_sovits_hifi.py \
    --pth checkpoints/<模型文件夹名>/model.pth \
    --config checkpoints/<模型文件夹名>/config.json \
    --out checkpoints/onnx_bundle/SoVits.onnx

# 6) 浅层扩散（已训练扩散模型时；导出后把 4 个 onnx 放入 bundle 的 diffusion/ 目录）
python diffusion/onnx_export.py --project-name <扩散模型名> \
    --model-path logs/44k/diffusion/model_188000.pt \
    --out-dir checkpoints/huawu_diff_188000

# 7) 汇总常量与配置（mel 滤波器组 / 重采样卷积核 / 扩散系数 / config.json）
python tools/make_onnx_bundle.py
```

导出结果位于 `checkpoints/onnx_bundle/`，目标机只需三个依赖：

```bash
pip install onnxruntime numpy soundfile
cd checkpoints/onnx_bundle
python infer.py -i input.wav -o output.wav                      # 主模型
python infer.py -i input.wav -o output.wav --shallow-diffusion  # + 浅层扩散
```

> 输入需为 44.1kHz；`--chunk 30` 长音频切片、`-t 2` 变调、`--seed` 固定随机、
> `--highpass 20` 输出高通（默认开启，去除直流/次声）。

数值对齐校验（需在 PyTorch 环境所在机器运行，逐组件比对编码器/F0/扩散/声码器/SoVits）：

```bash
python tools/verify_onnx_bundle.py
```

详细文件清单、输入输出规格与已知差异见 [`checkpoints/onnx_bundle/README.md`](checkpoints/onnx_bundle/README.md)，
选型与导出报告见 [`doc/model_selection_report.md`](doc/model_selection_report.md)。

## ⚙️ XPU设备训练建议

> ⚠️ **重要：当前建议使用 FP32 训练**

虽然 Intel XPU 硬件在底层支持 FP16/BF16 计算，但本项目在实际训练中验证发现，**混合精度（FP16/BF16）存在明显问题，训练不稳定**，因此**强烈建议当前仍使用 FP32（`fp16_run: false`）进行训练**。

**FP16 训练存在的问题**：
- FP16 指数范围仅 ±6.5×10⁴，梯度容易**下溢/溢出**，训练早期极易出现 `NaN` 损失并导致训练中断；
- 需要依赖 `GradScaler` 动态梯度缩放，但 PyTorch XPU 后端对 `GradScaler` 的支持不完善，`unscale_` 阶段可能出现 FP64 相关报错（见 [`train.py`](train.py:549)）；
- 在 Intel Arc A770 等设备上，FP16 满载训练更容易触发 `DEVICE_LOST` 崩溃。

**BF16 训练存在的问题**：
- BF16 尾数仅 7 位（约 2~3 位十进制精度），STFT/Mel 频谱计算精度损失严重，**输出音频会出现电子杂音**（[`modules/mel_processing.py`](modules/mel_processing.py:61) 已为此保留 FP32）；
- 损失函数（如 MSE）在 BF16 下精度损失明显，代码中已显式转回 FP32 规避（[`modules/losses.py`](modules/losses.py:8)）；
- `fused` 优化器在 BF16 下存在 FP64 兼容性问题，已被禁用（[`train.py`](train.py:266)）。

**结论**：FP16/BF16 带来的速度收益有限，但稳定性与音质损失代价较大，**当前请使用 FP32 训练**。

1. **推荐配置（FP32，稳定首选）**:
   ```json
   {
     "train": {
       "batch_size": 4,
       "fp16_run": false,
       "half_type": "fp32",
       "grad_accumulation_steps": 4,
       "all_in_mem": false
     }
   }
   ```

2. **精度支持检测**:
   运行以下脚本快速检测您的XPU设备精度支持情况（仅供了解硬件能力）：
   ```bash
   python check_xpu_precision.py
   ```

3. **性能优化建议**:
   - **优先 FP32**: 稳定优先，避免混合精度导致的 NaN 与电子杂音
   - **合理batch_size**: 根据显存调整，通常 4-8 之间
   - **梯度累积**: 使用 grad_accumulation_steps 模拟更大 batch_size
   - **内存管理**: 禁用 all_in_mem 避免内存溢出
   - **定期清理**: 训练中定期调用 torch.xpu.empty_cache()

4. **故障排除**:
   - 如果遇到训练不稳定，请确认使用 FP32（`fp16_run: false`）
   - 监控显存使用，适当调整 batch_size 和 grad_accumulation_steps
   - 确保驱动程序和PyTorch XPU版本为最新
   - 查看训练日志中的精度检测信息

## 🛑 已知问题

1. 在Ubuntu等Linux系统下，模型训练会出现显存溢出的情况，致使模型无法正常训练。相较于在Windows下进行训练，在Linux下训练时请将batch_size调小。
2. ✅ **已解决**：生成hubert与f0功能使用多线程配置生成预处理文件导致训练闪退的问题。现已在 [`preprocess_hubert_f0.py`](preprocess_hubert_f0.py) 中加入多项保护机制——自动限制每进程 PyTorch 线程数为 1、设置 `OMP_NUM_THREADS`/`MKL_NUM_THREADS`、自动检测 Intel Arc A770 并限制进程数或建议 CPU 模式、按 CPU 核心数与内存给出建议进程数，可放心使用 `--num_processes` 多线程预处理。
3. webUI.py基本不可用，运行会出现浏览器无限加载的情况。

## 🔗 参考项目及文献

+ [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
+ [AI知识库/语音合成So-VITS-SVC](https://geekdaxue.co/read/gptcn@aigc/ilBnT3M8EKeVKeH-)
+ [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 
+ [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
+ [PyTorch 在Intel GPU上入门](https://docs.pytorch.ac.cn/docs/stable/notes/get_start_xpu.html)
+ [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

---
<!-- English -->

# SoftVC VITS Singing Voice Conversion For Intel XPU

![wmm](./doc/img/1701608234384.png)

📻 A training and inference framework based on the So-VITS-SVC model, adapted to support Intel GPUs.

## ⚠️ Notes

1. This project **only supports Intel discrete/Integrated Graphics(XPU)**, CUDA support has been completely removed, do not use on NVIDIA or other platforms;
2. This project is an open-source, offline project that **cannot collect any user information or acquire user input data** and assumes no responsibility for any user input. This project **does not provide any form of support to any organization or individual**, so all AI models based on this project and synthesized audio **are unrelated to the contributors of this project**. All problems caused by this shall be **borne by the user**;
3. This project is only a framework project with no models, and any redistributions of the project are unrelated to the contributors of this project;
4. Please resolve dataset licensing issues on your own, **prohibited from using unlicensed datasets for training**. Any problems caused by using unlicensed datasets for training, the **full responsibility and consequences must be borne by the user**.

## 📗 Project Introduction

This project is based on the [So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc) project, the original project version is `4.1-Stable`, using PyTorch+XPU, adapted for Intel graphics cards. Used for voice tone conversion, AI covers, and other functions. Extracts source audio speech features through the SoftVC content encoder.

## 🚗 Supported Intel GPU Hardware

+ Intel Iris Xe Graphics eligible
+ Intel Arc A380 Graphics Card
+ Intel Arc A770 Graphics Card

## 🧪 Environment Configuration

### 1. Install Python Environment

This project uses Python 3.11, theoretically supports higher Python versions, but has not been tested yet.  
Since PyTorch+XPU requires a minimum of Python 3.10, you need to install Python 3.10 or above.
```bash
# Windows
winget install --id Python.Python.3.11

# Ubuntu
sudo apt install python3.11 python3.11-dev python3.11-venv

# Fedora
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. Create Virtual Environment

Execute terminal commands in the project root directory to create a virtual environment
```bash
# Windows
py -3.11 -m venv venv

# Linux
python3.11 -m venv venv
```

Activate the virtual environment
```bash
# Windows
venv\Scripts\Activate.ps1

# Linux
source venv/bin/activate
```

### 3. Install Project Dependencies

**Install PyTorch**

Current PyTorch officially supports Intel graphics cards, so just install PyTorch, no need to install IPEX anymore.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Download slowly or frequently terminated, you can use mirror sources
# Nanjing University
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/xpu
# Shanghai Jiao Tong University
pip install torch torchvision torchaudio --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/xpu
```

**Install Other Dependencies**

```bash
pip install -r requirements.txt
```

## 🔨 Pre-downloaded Model Files

### Encoder

Select one of the following encoders to use
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### 1. If using contentvec as audio encoder (recommended)

vec768l12 and vec256l9 require this encoder, please download from the following two links (**choose one**).

+ [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
+ [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)

Rename the file to `checkpoint_best_legacy_500.pt` and place it in the `pretrain` directory.

#### 2. If using hubertsoft as audio encoder

Download the model [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt).  
Place it in the `pretrain` directory.

#### 3. If using Whisper-ppg as audio encoder

+ Download the model [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), this model fits `whisper-ppg`.
+ Download the model [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), this model fits `whisper-ppg-large`.

Place them in the `pretrain` directory.

#### 4. If using cnhubertlarge as audio encoder

Download the model [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt).  
Place it in the `pretrain` directory.

#### 5. If using dphubert as audio encoder

Download the model [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth).  
Place it in the `pretrain` directory.

#### 6. If using WavLM as audio encoder

Download the model [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), this model fits `wavlmbase+`.  
Place it in the `pretrain` directory.

#### 7. If using OnnxHubert/ContentVec as audio encoder

Download the model [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main).  
Place it in the `pretrain` directory.

### Pre-trained Base Models (Optional)

Using pre-trained base models can achieve faster training speed and better training results.

+ Pre-trained base model files: `G_0.pth` `D_0.pth`
  + Place in `logs/44k` directory

+ Diffusion model pre-trained base model files: `model_0.pt`
  + Place in `logs/44k/diffusion` directory
  
SoVits base model files: [ms903/sovits4.0-768vec-layer12 at main](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/tree/main/sovits_768l12_pre_large_320k)  

The diffusion model references the Diffusion Model from [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC), the base model is compatible with the diffusion model base model from [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC), you can go to [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) to get the base model for the diffusion model.

### NSF-HIFIGAN (Optional)

If using the **NSF-HIFIGAN Enhancer** or **shallow diffusion**, you need to download the pre-trained NSF-HIFIGAN model.  
Pre-trained NSF-HIFIGAN vocoder: [nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip).  
After extracting, place the four files in the `pretrain/nsf_hifigan` directory.

### RMVPE (Optional)

If using the rmvpeF0 predictor, you need to download the pre-trained RMVPE model.  
Download the model [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), extract `rmvpe.zip`, rename the `model.pt` file inside to `rmvpe.pt` and place it in the `pretrain` directory.

## 📚 Training Data Preparation

Prepare several segments of single-person vocals without background music as training data and save them as wav files.  
It is recommended that the training audio contains both singing and ordinary speech audio.  
You can use [UVR5](https://github.com/Anjok07/ultimatevocalremovergui/) for vocal extraction work.

### 1. Audio Slicing

Slice the training audio, you can use [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer).

Organize the sliced audio according to the following file structure and put the dataset in the `dataset_raw` directory.
```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
There are no restrictions on the format of each audio file name, but for convenience later, it is recommended to use English naming for all.  
However, the file format must be wav.  
Also, custom speaker names can be used.
```
dataset_raw
└───HuaWuNyako
    ├───1.wav
    ├───a.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

### 2. Resample to 44100Hz Mono

```
python resample.py
```

Note: Although this project has resampling, mono conversion and loudness matching scripts resample.py, the default loudness matching is matched to 0db. This may cause damage to the audio quality. And python's loudness matching package pyloudnorm cannot limit the level, which will cause clipping. So it is recommended to consider using professional audio processing software such as Adobe Audition to do loudness matching.  
If loudness matching has already been done with other software, you can add --skip_loudnorm to skip the loudness matching step when running the above command.
```bash
python resample.py --skip_loudnorm
```

### 3. Automatically Split Training Set, Validation Set, and Auto-generate Configuration File

```
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder parameter options:

+ vec768l12 (default)
+ vec256l9
+ hubertsoft
+ whisper-ppg
+ whisper-ppg-large
+ cnhubertlarge
+ dphubert
+ wavlmbase+

If using loudness embedding, add the --vol_aug parameter.
```
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

After using this, the trained model will match the input source loudness, otherwise it will match the training set loudness.

#### Configuration File

At this point you can modify some parameters in the generated `config.json` and `diffusion.yaml`

**config.json**

+ keep_ckpts: Number of models to keep during training, 0 means keep all; default is to keep the last 3.
+ all_in_mem: Load all datasets into memory, can be enabled when disk IO on some platforms is too low and memory capacity is much larger than dataset size.
+ batch_size: Amount of data loaded to the GPU for a single training session, adjust to be below the memory capacity.
+ c_mel: Mel spectrogram loss weight, default value 45; controls the importance of spectral domain reconstruction error.
+ c_kl: KL divergence loss weight, default value 1.0; controls the regularization strength of the variational autoencoder.
+ c_fm: Feature matching loss weight, default value 0.5; controls the influence of feature matching loss in adversarial training. This parameter helps the generator learn intermediate feature representations closer to real data. Recommended value 0.1-1.0; too high may cause training instability.
+ vocoder_name: Select a vocoder; default is nsf-hifigan.
  + Vocoder list
    + nsf-hifigan
    + nsf-snake-hifigan


**diffusion.yaml**

+ cache_all_data: Load all datasets into memory, can be enabled when disk IO on some platforms is too low and memory capacity is much larger than dataset size
+ duration: Training audio slice duration, can be adjusted according to memory size, note that this value must be less than the shortest duration in the training set!
+ batch_size: Amount of data loaded to the GPU for a single training session, adjust to be below the memory capacity
+ timesteps: Total steps of the diffusion model, default is 1000.
+ k_step_max: Training can only train k_step_max steps of diffusion to save training time, note that this value must be less than timesteps, 0 means training the entire diffusion model, note that if you don't train the entire diffusion model you will not be able to use diffusion-only model inference!

### 4. Generate hubert and f0

```
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor optional parameters
+ crepe
+ dio
+ pm
+ harvest
+ rmvpe
+ fcpe

If the training set is too noisy, it is recommended to use crepe to process f0.  
If the f0_predictor parameter is omitted, the default value is rmvpe.

If shallow diffusion function is needed, add the --use_diff parameter.
```
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

#### 💻 Using CPU for Preprocessing

If you want to use CPU instead of XPU/GPU, you can specify it with the `--device` parameter:

```bash
# Force CPU mode
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu
```

**Advantages of CPU Mode**:
- ✅ Completely avoids XPU multiprocessing issues
- ✅ Can use more processes (recommended 4-8)
- ✅ More stable, won't cause system freeze
- ✅ Suitable for systems without dedicated GPU or insufficient VRAM

**CPU Mode Recommended Configurations**:

```bash
# 4-core CPU: use 2-3 processes
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 3

# 6-core CPU: use 4-5 processes (recommended)
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 4

# 8-core CPU: use 6 processes
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 6

# 16+ cores: use up to 12-16 processes (requires large memory)
python preprocess_hubert_f0.py --f0_predictor rmvpe --device cpu --num_processes 12
```

**Automatic Optimization**: Code automatically detects your CPU cores and memory size to calculate optimal process count and provides recommendations.

**User Control**: The program does NOT enforce limits. You can set any value you want. If it exceeds system capacity, detailed warnings will be displayed.

**Memory Usage Reference**:
- Without `--use_diff`: ~2-2.5GB per process
- With `--use_diff`: ~3-4GB per process

**Recommendation**: Set CPU process count to 75% of physical cores to balance speed and system responsiveness.

#### ⚠️ Important: Intel Arc A770 Multiprocessing Warning

**Known Issue**: According to Intel's official OpenVINO 2025.2 release notes, running multiple processes simultaneously on Intel Arc A770 may lead to system hangs.

**Technical Reasons**:
- Intel Arc A770 driver has stability issues with concurrent multi-process access
- Each process loading large models consumes significant memory and VRAM
- Resource contention between PyTorch threads and multiprocessing may cause deadlocks

**Recommended Configuration Options**:

##### Option 1: Maximum Stability (Strongly Recommended)
```bash
# Single process mode, most stable, suitable for most users
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 1
```
- ✅ Completely avoids multiprocessing conflicts
- ✅ Lowest memory usage
- ✅ Will not cause system freeze
- ⚠️ Relatively slower, but safest

##### Option 2: Moderate Acceleration (Arc A770 Users)
```bash
# 2-4 processes, balance between speed and stability
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 2
```
- ✅ 1.5-2x faster than single process
- ⚠️ Need to monitor system temperature and memory
- ⚠️ Not recommended to exceed 4 processes
- ❌ Still has small probability of causing system instability

##### Option 3: CPU Multi-core Acceleration (Non-XPU Users)
```bash
# CPU users can increase processes moderately, but still need limits
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 4
```
- ✅ Fully utilize CPU multi-core
- ⚠️ Recommended to set to half of CPU cores
- ⚠️ Maximum 8 processes

**Automatic Protection Mechanisms**:
- ✅ Code automatically detects Arc A770 and issues warnings
- ✅ Automatically limits Arc A770 to maximum 4 processes
- ✅ Limits PyTorch threads per process to 1
- ✅ Sets OMP_NUM_THREADS and MKL_NUM_THREADS environment variables
- ✅ Detailed logging output for monitoring progress

**Monitoring Recommendations**:
1. Open Task Manager to monitor CPU and memory usage
2. If memory usage exceeds 80%, stop immediately and reduce process count
3. Monitor system temperature to ensure proper cooling
4. For first-time use, start with `--num_processes=1` for testing

After completing the above steps, the `dataset` directory will contain the preprocessed data, and the `dataset_raw` folder can be deleted at this time.

### 5. Start Training

```
python train.py -c configs/config.json -m 44k
```

During training, TensorBoard logs will be written to the `logs/44k` directory, which can be viewed with the following command:

```bash
tensorboard --logdir logs/44k
```

To interrupt training, press `Ctrl+C` to terminate the training process.

Training checkpoints will be saved in the `logs/44k` directory, including:
- `G_*.pth`: Generator model checkpoints
- `D_*.pth`: Discriminator model checkpoints

Shallow diffusion model training
```
# Note: First preprocess data with --use_diff:
# python preprocess_hubert_f0.py --f0_predictor dio --use_diff

# Then train diffusion model separately
python train_diff.py -c configs/diffusion.yaml
```

If training is unstable and frequently interrupted, you can use `supervisor` for process guardian training to prevent model training interruption.  
Install supervisor
```bash
# Ubuntu
sudo apt install supervisor

# Fedora
sudo dnf install supervisor

# Windows
pip install supervisor-win
```

Model training
```
# Main model training
supervisord -n -c train_supervisord.conf

# Shallow diffusion model training
supervisord -n -c train_supervisord_diff.conf
```

You can use TensorBoard to monitor training status
```
tensorboard --logdir logs/44k --bind_all
```

After model training is completed, the model files are saved in the `logs/44k` directory, and the diffusion model is in `logs/44k/diffusion`

## 📝 Model Inference

### 1. Real-time Voice Changer (API Service)

Start the real-time voice changing API service (works with a VST plugin, listens on `0.0.0.0:6842` by default):
```bash
python flask_api.py
```
> Note: This project does not provide a `gui.py` GUI. Before use, edit `model_name`, `config_name`, and `cluster_model_path` in [`flask_api.py`](flask_api.py) to point to your model files.

### 2. Batch Inference

Use command line for batch inference:
```bash
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi"
```

**More common inference command examples**:

```bash
# 1. Pitch shift: raise by 2 semitones (-t 2), use rmvpe F0 predictor (-f0p rmvpe, more stable for songs)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 2 -s "buyizi" -f0p rmvpe

# 2. Voice conversion: enable automatic pitch prediction (-a, speech only; causes severe drift for singing)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "speech.wav" -t 0 -s "buyizi" -a

# 3. Clustering timbre control: use clustering model with ratio 0.5 (-cr 0.5, requires cluster/train_cluster.py first)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -cm "logs/44k/kmeans_10000.pt" -cr 0.5

# 4. Feature retrieval: use feature retrieval index with ratio 0.5 (-fr, requires train_index.py first)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -fr -cr 0.5

# 5. Shallow diffusion: reduce electronic artifacts (-shd, requires a trained diffusion model; must specify -dm/-dc diffusion model and config paths)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -dm "logs/44k/diffusion/model_0.pt" -dc "logs/44k/diffusion/config.yaml" -shd -ks 100 -vd cpu

# 6. Pure diffusion: diffusion-only inference (-od, requires a fully trained diffusion model; must specify -dm/-dc)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -dm "logs/44k/diffusion/model_0.pt" -dc "logs/44k/diffusion/config.yaml" -od -ks 100

# 7. Enhancer: enable NSF-HIFIGAN enhancer (-eh, may improve quality for models with little training data)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -eh

# 8. Speaker mix: enable character fusion (-usm, configure spkmix.py first)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -usm

# 9. Specify device: run both model and vocoder on CPU (-d cpu -vd cpu)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -d cpu -vd cpu

# 10. Batch inference: convert multiple files at once (-n accepts multiple filenames, -t maps one by one)
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "song1.wav" "song2.wav" -t 0 0 -s "buyizi"
```

Required parameters:
- `-m` | `--model_path`: Model path
- `-c` | `--config_path`: Configuration file path
- `-n` | `--clean_names`: Input audio filename(s), placed in raw directory
- `-t` | `--trans`: Pitch adjustment in semitones (supports positive and negative values)
- `-s` | `--spk_list`: Target speaker name
- `-cl` | `--clip`: Audio forced slicing duration in seconds, default 0 for automatic slicing

Optional parameters:
- `-lg` | `--linear_gradient`: Cross-fade duration between audio slices in seconds, adjust if vocal discontinuity occurs after forced slicing (default: 0)
- `-f0p` | `--f0_predictor`: F0 predictor selection (options: crepe, pm, dio, harvest, rmvpe, fcpe; default: pm). Note: crepe uses mean filtering for original F0
- `-a` | `--auto_predict_f0`: Enable automatic pitch prediction for voice conversion (not recommended for singing conversion as it may cause severe pitch drift)
- `-cm` | `--cluster_model_path`: Clustering model or feature retrieval index path. Leave empty to use default path of each scheme
- `-cr` | `--cluster_infer_ratio`: Clustering or feature retrieval ratio (range: 0-1). Set to 0 if no clustering model or feature retrieval is trained
- `-eh` | `--enhance`: Enable NSF_HIFIGAN enhancer. This option may improve audio quality for models with limited training data but may degrade quality for well-trained models (default: disabled)
- `-shd` | `--shallow_diffusion`: Enable shallow diffusion to address electronic music artifacts (default: disabled). Note: NSF_HIFIGAN enhancer will be disabled when this option is enabled
- `-usm` | `--use_spk_mix`: Enable character fusion/dynamic voice blending
- `-lea` | `--loudness_envelope_adjustment`: Loudness envelope mixing ratio between input source and output. Values closer to 1 use more of the output loudness envelope
- `-fr` | `--feature_retrieval`: Enable feature retrieval (disables clustering model). When enabled, cm and cr parameters become feature retrieval index path and mixing ratio respectively
- `-d` | `--device`: Main inference device. Auto-selected when not specified (Intel environment defaults to `xpu`)
- `-vd` | `--vocoder_device`: Vocoder (NSF-HiFiGAN) synthesis device, can be set to `cpu` or `xpu`. Defaults to follow the main inference device when not specified

Shallow diffusion settings:
+ `-dm` | `--diffusion_model_path`: Diffusion model path
+ `-dc` | `--diffusion_config_path`: Diffusion model configuration file path
+ `-ks` | `--k_step`: Number of diffusion steps. Higher values produce results closer to the diffusion model's output (default: 100)
+ `-od` | `--only_diffusion`: Pure diffusion mode. This mode will not load the SoVITS model and will perform inference using only the diffusion model
+ `-se` | `--second_encoding`: Secondary encoding. Performs additional encoding on the original audio before shallow diffusion. This is an experimental option with variable results

> **⚠️ XPU Vocoder High-Frequency Noise Issue (Important)**
> On Intel XPU, the `nsf_hifigan` vocoder's `noise_convs` (Conv1d) kernel has a numerical bug when processing low-amplitude harmonic signals (`har_source`), causing **white noise above 16 kHz** (measured: output 8k+ high-frequency ratio can exceed 30%, with overall weakened loudness).
> - **Symptom**: When shallow diffusion (`-shd`) is enabled, obvious high-frequency hissing/white noise appears in the output; pure SoVITS (no diffusion) is normal.
> - **Root cause**: Conv1d kernel defect in the PyTorch XPU backend (not a bug in this project's code; disabling oneDNN or input scaling cannot avoid it).
> - **Solution**: Specify the vocoder to synthesize on CPU during inference, while the diffusion model and main model stay on XPU, balancing quality and speed:
>   ```bash
>   python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi" -dm "logs/44k/diffusion/model_0.pt" -dc "logs/44k/diffusion/config.yaml" -shd -ks 100 -vd cpu
>   ```
>   With `-vd cpu`, the output 8k+ high-frequency ratio drops from 30%+ to <1%, and the audio quality returns to clean.

Note: When using whisper-ppg speech encoder for inference, set `--clip` to 25 and `--lg` to 1. Otherwise, normal inference will not be possible.

### f0 Predictor Comparison

The following is a comparison of the advantages and disadvantages of various f0 predictor algorithms during inference:
| Predictor |              Advantages              |                     Disadvantages                     |
| :-------: | :----------------------------------: | :---------------------------------------------------: |
|    pm     |      Fast speed, low resource usage      |                  Easy to produce mute sound                  |
|   crepe   |        Basically no mute sound        | High VRAM usage, comes with mean filtering, may cause pitch drift |
|    dio    |                  -                   |                    May cause pitch drift                    |
|  harvest  |     Better performance in low notes      |          Other pitch ranges are not as good as other algorithms          |
|   rmvpe   | Hexagon warrior, currently the most perfect predictor |     Almost no disadvantages (may have errors in extreme long low notes)     |

### Automatic f0 Prediction (Optional)

The 4.0 model training process will train an f0 predictor. For voice conversion, automatic pitch prediction can be enabled. If the effect is not good, manual prediction can also be used, but please do not enable this function when converting songs, it will cause **severe pitch drift**.  
Set `auto_predict_f0` to `true` in `inference_main`.

### Clustering Timbre Leakage Control (Optional)

The clustering approach can reduce timbre leakage, making the trained model more similar to the target timbre (but not particularly obvious), but pure clustering will reduce the model's articulation (causing unclear pronunciation) (this is very obvious). This model adopts a fusion approach, allowing linear control of the proportion of clustering vs non-clustering approaches, meaning you can manually adjust the ratio between "target timbre similarity" and "clear articulation" to find a suitable compromise.  
Using clustering does not require any changes to the previous steps, just train an additional clustering model. Although the effect is somewhat limited, the training cost is also relatively low.

Training:
```bash
python cluster/train_cluster.py
```
> Execute `python cluster/train_cluster.py`, the model output will be in `logs/44k/kmeans_10000.pt`  
> The clustering model can currently use gpu for training, execute `python cluster/train_cluster.py --gpu`

Inference:
> Specify `cluster_model_path` as the model output file in `inference_main.py`, leave blank to default to `logs/44k/kmeans_10000.pt`  
> Specify `cluster_infer_ratio` in `inference_main.py`, `0` means completely not using clustering, `1` means only using clustering, usually set to `0.5`

### Feature Retrieval

Like the clustering approach, it can reduce timbre leakage, and the articulation is slightly better than clustering, but it will reduce inference speed. It adopts a fusion approach that can linearly control the proportion of feature retrieval vs non-feature retrieval.

Training:

First, you need to execute after generating hubert and f0:
```bash
python train_index.py -c configs/config.json
```

The model output will be in `logs/44k/feature_and_index.pkl`

Inference:
> Need to first specify `--feature_retrieval`, at this time the clustering approach will automatically switch to the feature retrieval approach  
> Specify `cluster_model_path` as the model output file in `inference_main.py`, leave blank to default to `logs/44k/feature_and_index.pkl`  
> Specify `cluster_infer_ratio` in `inference_main.py`, `0` means completely not using feature retrieval, `1` means only using feature retrieval, usually set to `0.5`

## 📦 Model Compression

Remove training information from the model to reduce file size (approximately 1/3 of original size):

```
python compress_model.py -c="configs/config.json" -i="logs/44k/G_<model_name>.pth" -o="logs/44k/release.pth"
```

## 📤 ONNX Export

Export the model to ONNX format for deployment:

1. Create a `checkpoints/<model_folder_name>` folder in the project root;
2. Rename your trained model to `model.pth` and the config file to `config.json`, then place them in this folder;
3. Run the export command (`-n` specifies the model folder name):

```bash
python onnx_export.py -n <model_folder_name>
```

After exporting, `<model_folder_name>_SoVits.onnx` will be generated under `checkpoints/<model_folder_name>/`, and the corresponding MoeVS config `<model_folder_name>.json` under `checkpoints/`.

> Note: This project uses [`onnx_export.py`](onnx_export.py) to export (with speaker-mix support), not `export_onnx.py`.

### Full export (encoder / vocoder / F0) for cross-machine deployment

`onnx_export.py` exports only the SoVits body. To run inference on a machine **without a PyTorch
environment**, the content encoder, vocoder and F0 predictor must be exported as well. The scripts
below export all of them and assemble a self-contained inference bundle:

```bash
# 1) main model (for MoeVoiceStudio, 7 inputs) — put model.pth/config.json into checkpoints/<name>/ first
python onnx_export.py -n <model_folder_name>

# 2) content encoder ContentVec768L12 (exported from the original fairseq sub-graph)
python tools/export_onnx_encoder.py --out checkpoints/onnx_bundle/vec768l12.onnx

# 3) F0 predictor RMVPE (+ log-mel filterbank constants)
python tools/export_onnx_rmvpe.py --out checkpoints/onnx_bundle/rmvpe.onnx

# 4) vocoder NSF-HiFiGAN (excitation randomness externalized for reproducibility)
python tools/export_onnx_vocoder.py --out checkpoints/onnx_bundle/nsf_hifigan.onnx

# 5) SoVits, externalized-random build (fixes trace-frozen randomness and DC/subsonic artifacts)
python tools/export_onnx_sovits_hifi.py \
    --pth checkpoints/<model_folder_name>/model.pth \
    --config checkpoints/<model_folder_name>/config.json \
    --out checkpoints/onnx_bundle/SoVits.onnx

# 6) shallow diffusion (if trained; then copy the 4 onnx files into bundle/diffusion/)
python diffusion/onnx_export.py --project-name <diffusion_name> \
    --model-path logs/44k/diffusion/model_188000.pt --out-dir checkpoints/huawu_diff_188000

# 7) collect constants and config (mel filterbanks / resample kernels / diffusion coefficients)
python tools/make_onnx_bundle.py
```

The bundle is written to `checkpoints/onnx_bundle/`; the target machine only needs three packages:

```bash
pip install onnxruntime numpy soundfile
cd checkpoints/onnx_bundle
python infer.py -i input.wav -o output.wav                      # main model
python infer.py -i input.wav -o output.wav --shallow-diffusion  # + shallow diffusion
```

> 44.1 kHz input is required; `--chunk 30` for long audio, `-t 2` pitch shift, `--seed` for
> reproducibility, `--highpass 20` output high-pass (enabled by default to remove DC/subsonic).

Numerical parity check (run on the machine that has the PyTorch environment; compares encoder / F0 /
diffusion / vocoder / SoVits component by component):

```bash
python tools/verify_onnx_bundle.py
```

See [`checkpoints/onnx_bundle/README.md`](checkpoints/onnx_bundle/README.md) for the file list,
I/O specs and known differences.

## ⚙️ XPU Device Training Recommendations

> ⚠️ **Important: Use FP32 training for now**

Although Intel XPU hardware supports FP16/BF16 computation at the hardware level, actual training in this project has verified that **mixed precision (FP16/BF16) has notable stability issues**, so it is **strongly recommended to keep using FP32 (`fp16_run: false`) for training**.

**Problems with FP16 training**:
- FP16 has an exponent range of only ±6.5×10⁴, gradients easily **underflow/overflow**, and `NaN` losses are common early in training, interrupting it;
- It relies on `GradScaler` for dynamic gradient scaling, but PyTorch XPU backend support for `GradScaler` is incomplete, and `unscale_` may throw FP64-related errors (see [`train.py`](train.py:549));
- On devices such as Intel Arc A770, FP16 training at full load is more likely to trigger `DEVICE_LOST` crashes.

**Problems with BF16 training**:
- BF16 has only a 7-bit mantissa (about 2-3 significant decimal digits); STFT/Mel spectral computation suffers serious precision loss, causing **electronic noise in the output audio** ([`modules/mel_processing.py`](modules/mel_processing.py:61) already keeps FP32 for this reason);
- Loss functions (such as MSE) lose noticeable precision under BF16; the code already casts back to FP32 explicitly to avoid this ([`modules/losses.py`](modules/losses.py:8));
- The `fused` optimizer has FP64 compatibility issues under BF16 and has been disabled ([`train.py`](train.py:266)).

**Conclusion**: The speed gains of FP16/BF16 are limited, but the cost in stability and audio quality is high, so **please use FP32 for training now**.

1. **Recommended configuration (FP32, stable first)**:
   ```json
   {
     "train": {
       "batch_size": 4,
       "fp16_run": false,
       "half_type": "fp32",
       "grad_accumulation_steps": 4,
       "all_in_mem": false
     }
   }
   ```

2. **Precision support detection**:
   Run the following script to quickly check your XPU device's precision support (for understanding hardware capability only):
   ```bash
   python check_xpu_precision.py
   ```

3. **Performance optimization tips**:
   - **FP32 first**: prioritize stability and avoid NaN and electronic noise caused by mixed precision
   - **Reasonable batch_size**: adjust based on VRAM, usually 4-8
   - **Gradient accumulation**: use grad_accumulation_steps to simulate a larger batch_size
   - **Memory management**: disable all_in_mem to avoid running out of memory
   - **Periodic cleanup**: call torch.xpu.empty_cache() regularly during training

4. **Troubleshooting**:
   - If training is unstable, make sure FP32 is used (`fp16_run: false`)
   - Monitor VRAM usage and adjust batch_size and grad_accumulation_steps accordingly
   - Make sure the driver and PyTorch XPU version are up to date
   - Check the precision detection info in the training logs

## 🛑 Known Issues

1. Under Ubuntu and other Linux systems, model training will cause out-of-memory situations, causing the model to fail to train normally. Compared to training under Windows, when training under Linux, please reduce the batch_size.
2. ✅ **Resolved**: The crash during training caused by generating preprocessing files with multi-threading configuration. [`preprocess_hubert_f0.py`](preprocess_hubert_f0.py) now includes multiple protection mechanisms — automatically limiting PyTorch threads to 1 per process, setting `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, automatically detecting Intel Arc A770 and limiting process count or recommending CPU mode, and recommending a process count based on CPU cores and memory. You can safely use `--num_processes` for multi-threaded preprocessing.
3. webUI.py is basically unusable, running will cause the browser to load infinitely.

## 🔗 Reference Projects and Literature

+ [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
+ [AI Knowledge Base/Voice Synthesis So-VITS-SVC](https://geekdaxue.co/read/gptcn@aigc/ilBnT3M8EKeVKeH-)
+ [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 
+ [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
+ [Getting Started with PyTorch on Intel GPU](https://docs.pytorch.ac.cn/docs/stable/notes/get_start_xpu.html)
+ [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
