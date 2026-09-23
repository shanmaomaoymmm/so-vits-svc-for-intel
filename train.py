import logging
import multiprocessing
import os
import sys
import time

# XPU 环境变量必须在 import torch 之前设置！
# Intel Arc A770 优化
os.environ['NEOReadDebugKeys'] = '1'
os.environ['ClDeviceGlobalMemSizeAvailablePercent'] = '100'
os.environ['SYCL_CACHE_PERSISTENT'] = '1'
os.environ['ONEDEV_ENABLE_PROFILING'] = '0'
os.environ['ZE_AFFINITY_MASK'] = '0'
os.environ['LevelZeroEnableLargeBar'] = '1'
# 关键：限制线程数避免系统崩溃（2线程足够，不会耗尽内存）
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    CombinedDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 梯度裁剪监控（节流输出，避免每一步都刷屏）
# ---------------------------------------------------------------------------
_GRAD_REPORT = {"last_step": 0, "clip_count": 0, "peak": 0.0, "interval": 1000, "ratio": 10.0}


def _report_grad_clip(global_step, grad_norm_before, clip_value):
    """统计并节流上报梯度裁剪情况：每 interval 步汇总一次，而不是每步打印。"""
    st = _GRAD_REPORT
    if grad_norm_before > clip_value * st.get("ratio", 10.0):
        st["clip_count"] += 1
        st["peak"] = max(st["peak"], grad_norm_before)
    span = global_step - st["last_step"]
    if span >= st["interval"]:
        if st["clip_count"] > 0:
            print(f"[GRAD] step {global_step}: clipping active {st['clip_count']}/{span} steps, "
                  f"peak_before_clip={st['peak']:.1f} (clip={clip_value:g})")
        else:
            print(f"[GRAD] step {global_step}: no clipping in last {span} steps (clip={clip_value:g})")
        st["last_step"] = global_step
        st["clip_count"] = 0
        st["peak"] = 0.0

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def attempt_load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    """
    尝试加载单个检查点，如果加载失败返回False
    """
    try:
        print(f"Trying to load checkpoint: {checkpoint_path}")
        model, optimizer, learning_rate, epoch_str = utils.load_checkpoint(
            checkpoint_path, model, optimizer, skip_optimizer
        )
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
        return model, optimizer, learning_rate, epoch_str, True
    except Exception as e:
        print(f"Failed to load checkpoint: {checkpoint_path}, error: {str(e)}")
        return model, optimizer, 0, 0, False


def remap_discriminator_checkpoint(checkpoint_path):
    """
    将旧 MultiPeriodDiscriminator 检查点映射为 CombinedDiscriminator 格式。
    旧检查点键名: discriminators.0.convs...
    CombinedDiscriminator 期望: mpd.discriminators.0.convs...
    """
    import tempfile
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    saved_state_dict = checkpoint_dict['model']
    
    # 检查是否已有 mpd. 前缀（新格式），没有则需要添加
    if not any(k.startswith('mpd.') for k in saved_state_dict.keys()):
        print("  [REMAP] Adding 'mpd.' prefix to old discriminator checkpoint keys...")
        new_state_dict = {}
        for k, v in saved_state_dict.items():
            new_state_dict[f'mpd.{k}'] = v
        checkpoint_dict['model'] = new_state_dict
        
        # 写入临时文件供 load_checkpoint 使用
        tmp = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        torch.save(checkpoint_dict, tmp.name)
        tmp.close()
        print(f"  [REMAP] Remapped checkpoint saved to temp file")
        return tmp.name
    
    return checkpoint_path


def safe_load_latest_checkpoint(model_dir, model_name_pattern, model, optimizer=None, skip_optimizer=False):
    """
    安全地加载最新的可用检查点，如果最新检查点损坏则尝试次新的检查点
    """
    checkpoint_paths = utils.scan_checkpoint_paths(
        model_dir, model_name_pattern)

    if not checkpoint_paths:
        print(
            f"No checkpoints found for pattern {model_name_pattern} in {model_dir}")
        return model, optimizer, 0, 0

    # 按照迭代次数排序（从大到小）
    checkpoint_paths.sort(key=lambda f: int(
        "".join(filter(str.isdigit, f))), reverse=True)

    # 判别器检查点需要键名映射（旧 MPD → 新 CombinedDiscriminator）
    is_discriminator = 'D_' in model_name_pattern
    
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        load_path = checkpoint_path
        if is_discriminator:
            try:
                load_path = remap_discriminator_checkpoint(checkpoint_path)
            except Exception as e:
                print(f"  [REMAP] Failed to remap checkpoint: {e}")
        
        model, optimizer, learning_rate, epoch_str, success = attempt_load_checkpoint(
            load_path, model, optimizer, skip_optimizer
        )

        if success:
            # 成功加载后，更新全局步数
            global_step_name = checkpoint_path
            global_step_val = int(global_step_name[global_step_name.rfind(
                "_") + 1:global_step_name.rfind(".")]) + 1
            return model, optimizer, learning_rate, epoch_str, global_step_val

        # 如果这是最后一个检查点仍然失败，返回初始值
        if idx == len(checkpoint_paths) - 1:
            print("All checkpoints are corrupted, starting from scratch...")
            return model, optimizer, 0, 0, 0

    return model, optimizer, 0, 0, 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    # 检测设备类型 - 仅支持XPU设备
    # 环境变量已在文件顶部 import torch 之前设置
    if torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        device_type = 'xpu'
        print(f"XPU devices available: {n_gpus}, using XPU backend")
        print(f"XPU optimization enabled for Intel Arc A770 16GB with {multiprocessing.cpu_count()} CPU cores")
    else:
        raise RuntimeError(
            "No XPU device available. Training requires an Intel GPU.")

    assert n_gpus > 0, f"No GPU devices found. n_gpus: {n_gpus}"

    hps = utils.get_hparams()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, device_type))


def run(rank, n_gpus, hps, device_type):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(
            log_dir=os.path.join(hps.model_dir, "eval"))

    # 对于单GPU情况，不需要初始化分布式训练
    if n_gpus > 1:  # 只在多GPU时初始化分布式训练
        if device_type == 'xpu':
            backend = 'gloo' if os.name == 'nt' else 'ccl'
            init_method = 'tcp://127.0.0.1:12355'
            dist.init_process_group(
                backend=backend, init_method=init_method, world_size=n_gpus, rank=rank)
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}")
    else:
        # 单GPU模式，设置主设备
        if device_type == 'xpu':
            torch.xpu.set_device(rank)

    torch.manual_seed(hps.train.seed)

    if rank == 0:
        print(f"[DEBUG] Setting up DataLoader...")
    collate_fn = TextAudioCollate()
    # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    all_in_mem = hps.train.all_in_mem
    
    if rank == 0:
        print(f"[DEBUG] Creating dataset (all_in_mem={all_in_mem})...")
    train_dataset = TextAudioSpeakerLoader(
        hps.data.training_files, hps, all_in_mem=all_in_mem)
    if rank == 0:
        print(f"[DEBUG] Dataset created, size={len(train_dataset)}")

    # Intel Arc A770 16GB + 64GB RAM 优化配置
    cpu_count = multiprocessing.cpu_count()
    
    if all_in_mem:
        # 数据全部在内存时，减少worker数量避免冗余复制
        num_workers = 0  # 改为0避免多进程问题
        prefetch_factor = None
        pin_memory = False  # 数据已在内存，不需要pin
        persistent_workers = False
    else:
        # 磁盘加载模式：优化数据预取
        # A770有16GB显存，可以使用较多的worker和prefetch
        num_workers = min(4, cpu_count // 2)  # 根据CPU核心数动态调整
        prefetch_factor = 2
        pin_memory = True  # 启用pin memory加速H2D传输
        persistent_workers = os.name != 'nt'  # Linux下启用persistent workers提高性能

    # 允许通过 config 覆盖数据加载参数（num_workers<0 / null 表示沿用上面的自动策略）
    _cfg_nw = getattr(hps.train, 'num_workers', -1)
    if _cfg_nw is not None and int(_cfg_nw) >= 0:
        num_workers = int(_cfg_nw)
    _cfg_pin = getattr(hps.train, 'pin_memory', None)
    if _cfg_pin is not None:
        pin_memory = bool(_cfg_pin)
    _cfg_pf = getattr(hps.train, 'prefetch_factor', None)
    if _cfg_pf is not None:
        prefetch_factor = int(_cfg_pf)
    train_shuffle = bool(getattr(hps.train, 'train_shuffle', False))

    if rank == 0:
        print(f"[DEBUG] Creating DataLoader (workers={num_workers}, shuffle={train_shuffle})...")
    # 训练数据加载器配置
    train_loader_kwargs = {
        'dataset': train_dataset,
        'num_workers': num_workers,
        'shuffle': train_shuffle,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
        'batch_size': hps.train.batch_size,
        'collate_fn': collate_fn
    }
    
    # 只在num_workers > 0且非all_in_mem时设置prefetch_factor
    if num_workers > 0 and prefetch_factor is not None:
        train_loader_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    if rank == 0:
        print(f"[DEBUG] DataLoader created")

    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_fn
        )

    # 根据设备类型将模型移动到对应设备
    if device_type == 'xpu':
        device = torch.device(f'xpu:{rank}')
        # 注意：torch.xpu.set_per_process_memory_fraction 在当前版本不可用
        # XPU 会自动管理显存，无需手动设置
    else:
        raise RuntimeError(f"Unsupported device type: {device_type}")

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_d = CombinedDiscriminator(hps.model.use_spectral_norm).to(device)
    
    # XPU 优化：fused optimizer 在 BF16 下可能有 FP64 兼容性问题
    # 暂时禁用 fused，等待 PyTorch XPU 更新
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    # 对于单GPU，不需要使用DDP包装
    if n_gpus > 1:
        if torch.xpu.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)
    # 如果是单GPU，直接使用原始模型

    # 续训时是否恢复优化器状态（从 config 读取，避免每次换模型都改代码）
    skip_optimizer = bool(getattr(hps.train, 'skip_optimizer', False))
    # 判别器优化器：当 D 为完整 CombinedDiscriminator(MPD+MSD) 且优化器状态与参数一一对应时，
    # 应保持 false 以恢复 Adam 动量；仅在加载"只含 MPD"的旧检查点时才需置为 true。
    skip_optimizer_d = bool(getattr(hps.train, 'skip_optimizer_d', False))

    # 使用安全加载函数替代原有的加载逻辑
    try:
        net_g, optim_g, learning_rate_g, epoch_str, global_step_g = safe_load_latest_checkpoint(
            hps.model_dir, "G_*.pth", net_g, optim_g, skip_optimizer
        )
        # 判别器使用 skip_optimizer=True：
        # CombinedDiscriminator = MPD(旧) + MSD(新)，旧优化器状态缺少 MSD 参数
        # MPD 权重从检查点加载，MSD 随机初始化，优化器从头开始
        net_d, optim_d, learning_rate_d, epoch_str_d, global_step_d = safe_load_latest_checkpoint(
            hps.model_dir, "D_*.pth", net_d, optim_d, skip_optimizer_d
        )

        # 确保两个模型的epoch_str和global_step一致
        epoch_str = max(epoch_str, epoch_str_d, 1)

        # 如果global_step_g和global_step_d都大于0，使用较大的那个
        if global_step_g > 0 and global_step_d > 0:
            global_step = max(global_step_g, global_step_d)
        elif global_step_g > 0:
            global_step = global_step_g
        elif global_step_d > 0:
            global_step = global_step_d
        else:
            global_step = 0

    except Exception as e:
        print(f"Error during checkpoint loading: {str(e)}")
        epoch_str = 1
        global_step = 0

    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    if rank == 0:
        print(f"[DEBUG] Checkpoint loaded. epoch_str={epoch_str}, global_step={global_step}")
        print(f"[DEBUG] Creating schedulers...")

    # 从检查点恢复后，强制使用 config 中的 learning_rate
    # 原逻辑：optimizer.load_state_dict() 恢复了检查点里的 LR → config 的修改不生效
    if global_step > 0:
        for param_group in optim_g.param_groups:
            param_group['lr'] = hps.train.learning_rate
        for param_group in optim_d.param_groups:
            param_group['lr'] = hps.train.learning_rate
        if rank == 0:
            print(f"[FIX] Overrode optimizer LR from config: {hps.train.learning_rate}")

    warmup_epoch = hps.train.warmup_epochs
    # 学习率改为“按训练步衰减”(step-based)：
    #   每 lr_decay_steps 步将 lr 乘以 gamma(=lr_decay)，在 train_and_evaluate 内按步调用 step()。
    # ExponentialLR 的 last_epoch 固定为 -1，衰减完全由按步调用控制，
    # 避免原先“每 epoch 才衰减一次”导致的 12.7 万步仅衰减 3% 的问题。
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=-1)

    # ──────────────────────────────────────────────────────────────
    # 修复：消除学习率"过山车"效应
    # line 332-336 将 optimizer LR 覆盖为 config 值是为了确保
    # scheduler.base_lrs 取到正确的初始值，但副作用是当前 epoch
    # 将以初始 LR 训练（过高），造成 TensorBoard 上 LR 的跳变。
    #
    # 此处根据指数衰减公式计算当前 epoch 应有的 LR 并重新设置，
    # 使恢复后的第一个 epoch 也能使用正确的衰减值。
    # ──────────────────────────────────────────────────────────────
    if global_step > 0 and not skip_optimizer:
        # 检测用户是否修改了 config 中的 learning_rate
        # learning_rate_g 来自 checkpoint 元数据（保存时的 config LR）
        config_lr_changed = abs(hps.train.learning_rate - learning_rate_g) > 1e-12
        if not config_lr_changed:
            # step-based 衰减：已衰减次数 = global_step // lr_decay_steps
            lr_decay_steps = int(getattr(hps.train, 'lr_decay_steps', 0) or 0)
            decay_exponent = (global_step // lr_decay_steps) if lr_decay_steps > 0 else 0
            correct_lr = hps.train.learning_rate * (hps.train.lr_decay ** decay_exponent)
            for param_group in optim_g.param_groups:
                param_group['lr'] = correct_lr
            for param_group in optim_d.param_groups:
                param_group['lr'] = correct_lr
            if rank == 0:
                print(f"[FIX] Corrected LR for step {global_step}: "
                      f"{hps.train.learning_rate:.2e} * {hps.train.lr_decay}^{decay_exponent} = {correct_lr:.10f}")
        elif rank == 0:
            print(f"[FIX] Config LR changed ({learning_rate_g:.2e} -> {hps.train.learning_rate:.2e}), "
                  f"using new value directly (no decay correction)")

    if rank == 0:
        print(f"[DEBUG] Schedulers created")
        print(f"[DEBUG] Setting up GradScaler...")

    # 根据设备类型创建GradScaler
    if device_type == 'xpu':
        scaler = GradScaler(device_type, enabled=hps.train.fp16_run)
    else:
        # CPU模式下不使用GradScaler
        scaler = GradScaler("cpu", enabled=False)

    if rank == 0:
        print(f"[DEBUG] GradScaler created (fp16_run={hps.train.fp16_run})")
        print(f"[DEBUG] Starting training loop (epoch {epoch_str} to {hps.train.epochs})...")
        sys.stdout.flush()

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / \
                    warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / \
                    warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval], device_type)
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None, device_type)

        # 学习率衰减已改为按训练步在 train_and_evaluate 内执行，这里不再按 epoch 衰减。
        # 达到 max_steps 后停止训练。
        max_steps = int(getattr(hps.train, 'max_steps', 0) or 0)
        if max_steps > 0 and global_step >= max_steps:
            if rank == 0:
                logger.info(f"====> Reached max_steps={max_steps} (global_step={global_step}). Stop training.")
                try:
                    writer.flush()
                    writer.close()
                    writer_eval.flush()
                    writer_eval.close()
                except Exception as _e:
                    logger.info(f"[WARN] failed to close writers: {_e}")
            break


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, device_type):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16

    # step-based 学习率衰减与最大训练步数（从 config 读取）
    lr_decay_steps = int(getattr(hps.train, 'lr_decay_steps', 0) or 0)
    max_steps = int(getattr(hps.train, 'max_steps', 0) or 0)
    # 梯度裁剪日志的节流间隔与告警倍数（从 config 读取）
    _GRAD_REPORT["interval"] = int(getattr(hps.train, 'grad_warn_interval', 1000) or 1000)
    _GRAD_REPORT["ratio"] = float(getattr(hps.train, 'grad_warn_ratio', 10.0) or 10.0)
    
    # Intel XPU BF16 优化：BF16 不需要 GradScaler（数值范围与FP32相同）
    # BF16: ±3.4×10³⁸ vs FP16: ±6.5×10⁴
    use_grad_scaler = hps.train.fp16_run and half_type != torch.bfloat16
    
    if rank == 0 and hps.train.fp16_run and not use_grad_scaler:
        print("[OPTIMIZATION] BF16 mode detected - GradScaler disabled (not needed for BF16)")
        print("[OPTIMIZATION] BF16 has same exponent range as FP32, no gradient underflow risk")

    # 根据设备类型获取device
    if device_type == 'xpu':
        device = torch.device(f'xpu:{rank}')
    else:
        raise RuntimeError(f"Unsupported device type: {device_type}")

    # 在第一个epoch打印性能提示
    if epoch == 1 and rank == 0:
        print_xpu_performance_tips(rank, hps, device)
        # 验证模型确实在 XPU 上
        print("\n[DIAGNOSTIC] Verifying XPU execution:")
        print(f"  net_g device: {next(net_g.parameters()).device}")
        print(f"  net_d device: {next(net_d.parameters()).device}")
        print(f"  Expected: xpu:{rank}")
        # 关键断言：确保模型在 XPU 上
        assert str(next(net_g.parameters()).device).startswith('xpu'), \
            f"CRITICAL: net_g is on {next(net_g.parameters()).device}, expected XPU!"
        assert str(next(net_d.parameters()).device).startswith('xpu'), \
            f"CRITICAL: net_d is on {next(net_d.parameters()).device}, expected XPU!"
        print("  ✓ Model parameters on XPU")
        # 验证显存使用（如果只有几百MB说明没在XPU上跑）
        torch.xpu.synchronize(device)
        mem_mb = torch.xpu.memory_allocated(device) / 1024 / 1024
        print(f"  XPU memory allocated: {mem_mb:.0f} MB")
        if mem_mb < 500:
            print(f"  ⚠️ WARNING: Only {mem_mb:.0f}MB on XPU - model may not be training!")
        print("  ✓ XPU verification complete\n")

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # 获取梯度累积步数，以模拟更大的批次大小
    grad_accumulation_steps = getattr(hps.train, 'grad_accumulation_steps', 1)

    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv, volume = items

        # Intel XPU 优化：批量数据传输，减少PCIe传输次数
        if device_type == 'xpu':
            device = torch.device(f'xpu:{rank}')
            # 使用非阻塞传输重叠计算和数据传输
            g = spk.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            f0 = f0.to(device, non_blocking=True)
            uv = uv.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            volume = volume.to(device, non_blocking=True) if volume is not None else None
            
            # XPU 优化：仅在日志记录间隔同步，减少不必要的 stall
            if global_step % hps.train.log_interval == 0:
                torch.xpu.synchronize(device)
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}")
        
        # 预计算 mel 谱（在GPU上进行）
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        # Discriminator training
        y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                spec_lengths=lengths, vol=volume)

        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        y = commons.slice_segments(
            y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        with autocast(device_type=device_type, enabled=hps.train.fp16_run, dtype=half_type):
            with autocast(device_type=device_type, enabled=False, dtype=half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc / grad_accumulation_steps

        # 修复：zero_grad 必须在 backward 之前，清理上一轮 G 步残留的判别器梯度
        # 原位置在 optim_d.step() 之后导致残留 G 梯度污染本轮 D 梯度
        if (batch_idx) % grad_accumulation_steps == 0:
            optim_d.zero_grad()

        # Discriminator backward (梯度累积)
        if use_grad_scaler:
            scaler.scale(loss_disc_all).backward()
        else:
            loss_disc_all.backward()

        # 只在累积的最后一步更新参数
        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if use_grad_scaler:
                try:
                    scaler.unscale_(optim_d)
                except RuntimeError as e:
                    if "fp64" in str(e):
                        print(f"Warning: FP64 aspect error during unscale, skipping: {e}")
                    else:
                        raise e
                grad_clip_value_d = getattr(hps.train, 'grad_clip_d', getattr(hps.train, 'grad_clip', 5.0))
                # 先计算裁剪前的梯度范数用于监控
                grad_norm_d_before_clip = torch.nn.utils.clip_grad_norm_(net_d.parameters(), float('inf'))
                # 再进行实际的梯度裁剪
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), grad_clip_value_d)
                scaler.step(optim_d)
                scaler.update()
            else:
                grad_clip_value_d = getattr(hps.train, 'grad_clip_d', getattr(hps.train, 'grad_clip', 5.0))
                # 先计算裁剪前的梯度范数用于监控
                grad_norm_d_before_clip = torch.nn.utils.clip_grad_norm_(net_d.parameters(), float('inf'))
                # 再进行实际的梯度裁剪
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), grad_clip_value_d)
                optim_d.step()

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        with autocast(device_type=device_type, enabled=hps.train.fp16_run, dtype=half_type):
            with autocast(device_type=device_type, enabled=False, dtype=half_type):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                      z_mask) * hps.train.c_kl
                # 添加c_fm权重控制
                c_fm = getattr(hps.train, 'c_fm', 1.0)
                loss_fm = feature_loss(fmap_r, fmap_g) * c_fm
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # 修改：检查模型是否被DDP包装
                if hasattr(net_g, 'module'):
                    # DDP包装的模型
                    use_automatic_f0_prediction = net_g.module.use_automatic_f0_prediction
                else:
                    # 原始模型
                    use_automatic_f0_prediction = net_g.use_automatic_f0_prediction

                loss_lf0 = F.mse_loss(
                    pred_lf0, lf0) if use_automatic_f0_prediction else 0
                loss_gen_all = (loss_gen + loss_fm + loss_mel +
                                loss_kl + loss_lf0) / grad_accumulation_steps

        # Generator backward (梯度累积)
        if use_grad_scaler:
            scaler.scale(loss_gen_all).backward()
        else:
            loss_gen_all.backward()

        # 只在累积的最后一步更新参数
        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if use_grad_scaler:
                try:
                    scaler.unscale_(optim_g)
                except RuntimeError as e:
                    if "fp64" in str(e):
                        print(f"Warning: FP64 aspect error during unscale, skipping: {e}")
                    else:
                        raise e
                grad_clip_value_g = getattr(hps.train, 'grad_clip_g', getattr(hps.train, 'grad_clip', 1.0))
                # 先计算裁剪前的梯度范数用于监控
                grad_norm_g_before_clip = torch.nn.utils.clip_grad_norm_(net_g.parameters(), float('inf'))
                # 再进行实际的梯度裁剪
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), grad_clip_value_g)
                
                # 梯度裁剪监控（节流上报，避免每步刷屏）
                _report_grad_clip(global_step, float(grad_norm_g_before_clip), float(grad_clip_value_g))
                scaler.step(optim_g)
                scaler.update()
            else:
                grad_clip_value_g = getattr(hps.train, 'grad_clip_g', getattr(hps.train, 'grad_clip', 1.0))
                # 先计算裁剪前的梯度范数用于监控
                grad_norm_g_before_clip = torch.nn.utils.clip_grad_norm_(net_g.parameters(), float('inf'))
                # 再进行实际的梯度裁剪
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), grad_clip_value_g)
                
                # 梯度裁剪监控（节流上报，避免每步刷屏）
                _report_grad_clip(global_step, float(grad_norm_g_before_clip), float(grad_clip_value_g))
                optim_g.step()
            optim_g.zero_grad()

        # 梯度累积：只有在累积步数的最后一步才进行日志和验证
        if (batch_idx + 1) % grad_accumulation_steps != 0 and (batch_idx + 1) != len(train_loader):
            continue

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]

                # 检查是否有任何损失值为nan
                for i, loss in enumerate(losses):
                    if torch.isnan(loss):
                        raise ValueError(
                            f' [x] NaN loss detected at step {global_step} in loss {i}: {losses[i]}')

                reference_loss = 0
                for i in losses:
                    reference_loss += i
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(
                    f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all * grad_accumulation_steps, "loss/d/total": loss_disc_all * grad_accumulation_steps, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                # 修改：同样检查模型是否被DDP包装
                if hasattr(net_g, 'module'):
                    use_automatic_f0_prediction = net_g.module.use_automatic_f0_prediction
                else:
                    use_automatic_f0_prediction = net_g.use_automatic_f0_prediction

                if use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                            pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                 norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

                # Intel XPU 稳定性优化：定期清理显存缓存，
                # 缓解长时间满载训练导致的显存碎片积累与 DEVICE_LOST(UR_RESULT_ERROR_DEVICE_LOST)
                if device_type == 'xpu':
                    torch.xpu.empty_cache()

            if global_step % hps.train.eval_interval == 0:
                # 再次检查损失是否为nan，确保在保存检查点之前没有nan
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                has_nan = False
                for i, loss in enumerate(losses):
                    if torch.isnan(loss):
                        print(
                            f' [!] Skipping checkpoint save due to NaN in loss {i}: {losses[i]} at step {global_step}')
                        has_nan = True
                        break

                if not has_nan:
                    evaluate(hps, net_g, eval_loader, writer_eval, device_type)
                    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                    utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                    keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                    if keep_ckpts > 0:
                        utils.clean_checkpoints(
                            path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                else:
                    # 抛出异常以停止训练
                    raise ValueError(
                        ' [x] NaN loss detected, stopping training')

        global_step += 1

        # 按训练步衰减学习率：每 lr_decay_steps 步将 lr 乘以 gamma 一次
        if lr_decay_steps > 0 and global_step % lr_decay_steps == 0:
            scheduler_g.step()
            scheduler_d.step()
            if rank == 0:
                logger.info(f"[LR] Step {global_step}: lr decayed to "
                            f"{optim_g.param_groups[0]['lr']:.3e}")

        # 达到 max_steps 时保存最终检查点并结束本轮（由 run() 的 epoch 循环 break 退出）
        if max_steps > 0 and global_step >= max_steps:
            if rank == 0:
                logger.info(
                    f"====> Reached max_steps={max_steps}. Saving final checkpoint "
                    f"G_{global_step}.pth / D_{global_step}.pth")
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
            return

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval, device_type):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv, volume = items

            # 根据设备类型将数据移动到对应设备
            if device_type == 'xpu':
                device = torch.device('xpu:0')
            else:
                raise RuntimeError(f"Unsupported device type: {device_type}")

            g = spk[:1].to(device)
            spec, y = spec[:1].to(device), y[:1].to(device)
            c = c[:1].to(device)
            f0 = f0[:1].to(device)
            uv = uv[:1].to(device)
            if volume is not None:
                volume = volume[:1].to(device)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)

            # 修改：检查模型是否被DDP包装
            if hasattr(generator, 'module'):
                # DDP包装的模型
                y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)
            else:
                # 原始模型
                y_hat, _ = generator.infer(c, f0, uv, g=g, vol=volume)

            # 使用正确的设备类型进行autocast
            with autocast(device_type=device_type, enabled=False, dtype=half_type):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


def print_xpu_performance_tips(rank, hps, device):
    """打印Intel XPU性能优化提示"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("Intel Arc A770 16GB Training Optimization Tips")
        print("=" * 70)
        print(f"✓ Data Loading: {'all_in_mem' if hps.train.all_in_mem else 'disk-based'} mode")
        print(f"✓ Mixed Precision: {hps.train.half_type if hps.train.fp16_run else 'disabled'}")
        print(f"✓ Batch Size: {hps.train.batch_size}")
        
        # 显存使用情况
        try:
            mem_allocated = torch.xpu.memory_allocated(device) / 1024**3
            mem_reserved = torch.xpu.memory_reserved(device) / 1024**3
            print(f"✓ XPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        except:
            pass
        
        print("\nPerformance Recommendations:")
        print("  • For 16GB VRAM: batch_size 8-16 is optimal")
        print("  • Use BF16 for better stability on Intel XPU")
        print("  • Enable all_in_mem if dataset < 64GB RAM")
        print("  • Monitor with: intel_gpu_top -J")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
