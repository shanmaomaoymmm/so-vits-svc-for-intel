"""
TensorBoard 训练监控邮件脚本
============================
从 TensorBoard (http://127.0.0.1:6006) 获取训练数据，
生成图表并发送邮件报告。

依赖安装:
  pip install matplotlib

使用:
  python tb_email_monitor.py                    # 立即发送一次
  python tb_email_monitor.py --interval 3600    # 每 3600 秒发送一次
  python tb_email_monitor.py --interval 3600 --once  # 发送一次后退出

邮件配置:
  编辑 configs/email_config.json 文件（参照 configs/email_config.json.example）
  该文件已加入 .gitignore，不会提交到仓库
"""

import argparse
import io
import json
import os
import smtplib
import sys
import time
import urllib.request
import urllib.error
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("[ERROR] matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

# ============================================================
# 📋 配置加载
# ============================================================

# 项目根目录（本脚本位于 tools/ 下，因此向上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认邮件配置文件（相对项目根目录，不受当前工作目录影响）
CONFIG_FILE = PROJECT_ROOT / "configs" / "email_config.json"
EXAMPLE_FILE = PROJECT_ROOT / "configs" / "email_config.json.example"

def load_email_config():
    """从 configs/email_config.json 加载 SMTP 配置"""
    if not CONFIG_FILE.exists():
        print(f"[ERROR] 邮件配置文件不存在: {CONFIG_FILE}")
        print(f"  请复制 {EXAMPLE_FILE} 为 {CONFIG_FILE} 并填写 SMTP 配置")
        print(f"  示例: Copy-Item \"{EXAMPLE_FILE}\" \"{CONFIG_FILE}\"")
        sys.exit(1)
    with open(CONFIG_FILE, encoding='utf-8') as f:
        cfg = json.load(f)
    required_keys = ["host", "port", "username", "password", "from_addr", "to_addrs"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        print(f"[ERROR] 邮件配置缺少必要字段: {missing}")
        sys.exit(1)
    # 检查是否还是占位符值
    if "YOUR_" in str(cfg.get("password", "")):
        print(f"[ERROR] 邮件配置中的 password 仍是占位符值，请填写真实的 SMTP 授权码")
        print(f"  编辑文件: {CONFIG_FILE}")
        sys.exit(1)
    return cfg

# 全局 SMTP 配置（由 load_email_config() 初始化）
SMTP_CONFIG = None

# TensorBoard 地址
TENSORBOARD_URL = "http://127.0.0.1:6006"

# 临时文件目录
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


# ============================================================
# 🔌 TensorBoard API 数据获取
# ============================================================

def tb_request(path):
    """向 TensorBoard API 发送请求"""
    url = f"{TENSORBOARD_URL}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8')
    except urllib.error.URLError as e:
        raise ConnectionError(f"无法连接到 TensorBoard ({url}): {e}")
    except Exception as e:
        raise ConnectionError(f"请求 TensorBoard 失败 ({url}): {e}")


def tb_request_binary(path):
    """向 TensorBoard API 发送请求，返回二进制数据"""
    url = f"{TENSORBOARD_URL}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise ConnectionError(f"无法连接到 TensorBoard ({url}): {e}")
    except Exception as e:
        raise ConnectionError(f"请求 TensorBoard 失败 ({url}): {e}")


def get_scalar_data(tag, run="."):
    """获取标量数据，返回 [(step, value, wall_time), ...]"""
    raw = tb_request(f"/data/plugin/scalars/scalars?tag={urllib.parse.quote(tag)}&run={urllib.parse.quote(run)}")
    data = json.loads(raw)
    return [(item[1], item[2], item[0]) for item in data]  # (step, value, wall_time)


def get_latest_audio(tag, run="eval"):
    """获取最新的音频数据，返回 (step, wav_bytes) 或 None"""
    raw = tb_request(f"/data/plugin/audio/audio?tag={urllib.parse.quote(tag)}&run={urllib.parse.quote(run)}")
    items = json.loads(raw)
    if not items:
        return None
    
    # 找到 step 最大的（最新的）
    latest = max(items, key=lambda x: x['step'])
    step = latest['step']
    
    # 下载音频 - query 是完整查询参数，直接拼接
    query = latest.get('query', '')
    if query:
        try:
            wav_data = tb_request_binary(f"/data/plugin/audio/audio?{query}")
            return (step, wav_data)
        except Exception as e:
            print(f"     音频下载失败: {e}")
    
    return (step, None)


def get_latest_image(tag, run="."):
    """获取最新的图片，返回 (step, png_bytes) 或 None"""
    raw = tb_request(f"/data/plugin/images/images?tag={urllib.parse.quote(tag)}&run={urllib.parse.quote(run)}")
    items = json.loads(raw)
    if not items:
        return None
    
    latest = max(items, key=lambda x: x['step'])
    step = latest['step']
    
    query = latest.get('query', '')
    if query:
        try:
            img_data = tb_request_binary(f"/data/plugin/images/image?{query}")
            return (step, img_data)
        except:
            pass
    
    return (step, None)


def get_scalar_tags():
    """获取所有标量标签"""
    raw = tb_request("/data/plugin/scalars/tags")
    return json.loads(raw)


def get_audio_tags():
    """获取所有音频标签"""
    raw = tb_request("/data/plugin/audio/tags")
    return json.loads(raw)


# ============================================================
# 📊 图表生成
# ============================================================

# 图表样式配置
PLOT_STYLE = {
    'figure.figsize': (12, 5),
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
}

def make_scalar_chart(data_triples, title, ylabel, filename, log_scale=False):
    """
    生成标量曲线图 — TensorBoard Time Series 风格
    data_triples: [(step, value, wall_time), ...]
    - X轴为 Step
    - 底部显示累计训练时间
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    
    if not data_triples:
        return
    
    steps = [p[0] for p in data_triples]
    values = [p[1] for p in data_triples]
    wall_times = [p[2] for p in data_triples]
    x_data = steps
    
    # ====== 计算平滑线 ======
    if len(values) > 30:
        window = max(3, len(values) // 100)
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            end = min(len(values), i + window + 1)
            smoothed.append(sum(values[start:end]) / (end - start))
    else:
        smoothed = values
    
    # ====== 绘制：原始散点 + 平滑线 ======
    # 原始数据：半透明蓝线 + 散点
    ax.plot(x_data, values, color='#2196F3', linewidth=0.6, alpha=0.25, zorder=1)
    dot_step = max(1, len(x_data) // 150)
    ax.scatter(x_data[::dot_step], values[::dot_step], s=6,
               color='#2196F3', alpha=0.35, zorder=2, marker='.', linewidths=0)
    # 平滑线：橙色（用户喜欢的配色）
    ax.plot(x_data, smoothed, color='#FF5722', linewidth=2.2, alpha=0.9, zorder=3)
    
    # ====== Y轴范围：基于平滑数据（尖峰溢出） ======
    if not log_scale:
        s_min, s_max = min(smoothed), max(smoothed)
        if s_max > s_min:
            margin = (s_max - s_min) * 0.15
            ax.set_ylim(max(0, s_min - margin), s_max + margin)
    
    # ====== 右下角最新值标注 ======
    latest_val = values[-1]
    min_val = min(values)
    max_val = max(values)
    
    ax.annotate(f'{latest_val:.4f}', xy=(x_data[-1], smoothed[-1]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=11, fontfamily='monospace', color='#D32F2F', fontweight='bold',
                va='center',
                arrowprops=dict(arrowstyle='-', color='#D32F2F', lw=1.0))
    
    # ====== 底部信息栏（含累计训练时间） ======
    start_wt = wall_times[0]
    total_sec = wall_times[-1] - start_wt
    total_h = total_sec / 3600.0
    if total_h > 24:
        time_str = f'{total_h/24:.1f}d'
    else:
        time_str = f'{total_h:.1f}h'
    
    info_text = (f'Time {time_str}  |  '
                 f'Step {int(steps[-1])}  |  '
                 f'Latest: {latest_val:.4f}  |  '
                 f'Min: {min_val:.4f}  |  '
                 f'Max: {max_val:.4f}')
    
    ax.text(0.5, -0.28, info_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', color='#666',
            va='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa', edgecolor='#e0e0e0', lw=0.8))
    
    # ====== 格式化 ======
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Step', fontsize=10, color='#555')
    ax.set_ylabel(ylabel, fontsize=10, color='#555')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.2, color='#ccc', linestyle='-')
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ddd')
    ax.spines['bottom'].set_color('#ddd')
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(filename, dpi=100, bbox_inches='tight', format='jpeg')
    plt.close(fig)


def generate_all_charts():
    """
    生成三组独立图表（每组内每个指标一张独立子图）：
      组1 - 梯度: grad_norm_g, grad_norm_d
      组2 - 学习率: learning_rate
      组3 - 损失: loss/g/total, loss/g/mel, loss/g/fm, loss/g/kl, loss/d/total
    返回 [(group_name, tag, filepath), ...]
    """
    chart_files = []
    tags = get_scalar_tags()
    run = list(tags.keys())[0] if tags else "."
    
    print("  [获取标量数据]")
    
    data_cache = {}
    needed_tags = [
        "grad_norm_g", "grad_norm_d",
        "learning_rate",
        "loss/g/total", "loss/g/mel", "loss/g/fm", "loss/g/kl", "loss/d/total",
    ]
    for tag in needed_tags:
        try:
            data = get_scalar_data(tag, run)
            if data:
                data_cache[tag] = data
                print(f"    ✓ {tag} ({len(data)} data points)")
            else:
                print(f"    ✗ {tag} (no data)")
        except Exception as e:
            print(f"    ✗ {tag} ({e})")
    
    # ====== 组1: 梯度 ======
    print("  [生成] 梯度组...")
    grad_map = {
        "grad_norm_g": ("Generator Grad Norm", True),
        "grad_norm_d": ("Discriminator Grad Norm", True),
    }
    for tag, (title, log_s) in grad_map.items():
        if tag in data_cache:
            filename = TMP_DIR / f"chart_{tag.replace('/', '_')}.jpg"
            make_scalar_chart(data_cache[tag], title, "Gradient Norm", filename, log_s)
            chart_files.append(("gradient", tag, filename))
            print(f"    ✓ {tag}")
    
    # ====== 组2: 学习率 ======
    print("  [生成] 学习率图...")
    if "learning_rate" in data_cache:
        filename = TMP_DIR / "chart_learning_rate.jpg"
        make_scalar_chart(data_cache["learning_rate"], "Learning Rate", "LR", filename)
        chart_files.append(("lr", "learning_rate", filename))
        print("    ✓ learning_rate")
    
    # ====== 组3: 损失 ======
    print("  [生成] 损失组...")
    loss_map = {
        "loss/g/total": ("Generator Total Loss", False),
        "loss/g/mel": ("Generator Mel Loss", False),
        "loss/g/fm": ("Generator FM Loss", False),
        "loss/g/kl": ("Generator KL Loss", False),
        "loss/d/total": ("Discriminator Total Loss", False),
    }
    for tag, (title, _) in loss_map.items():
        if tag in data_cache:
            filename = TMP_DIR / f"chart_{tag.replace('/', '_')}.jpg"
            make_scalar_chart(data_cache[tag], title, "Loss", filename)
            chart_files.append(("loss", tag, filename))
            print(f"    ✓ {tag}")
    
    return chart_files


def generate_latest_values_table():
    """生成最新数值的 HTML 表格"""
    tags = get_scalar_tags()
    run = list(tags.keys())[0] if tags else "."
    
    metrics = [
        "loss/g/total", "loss/g/mel", "loss/g/fm", "loss/g/kl",
        "loss/d/total", "learning_rate",
        "grad_norm_g", "grad_norm_d",
    ]
    
    rows = []
    for tag in metrics:
        try:
            data = get_scalar_data(tag, run)
            if data:
                latest_step, latest_val, _ = data[-1]
                # 取最近 100 个点的均值
                recent = data[-min(100, len(data)):]
                avg_val = sum(v for _, v, _ in recent) / len(recent)
                
                # 取整体范围
                all_vals = [v for _, v, _ in data]
                min_val, max_val = min(all_vals), max(all_vals)
                
                name = tag.replace("loss/g/", "G ").replace("loss/d/", "D ")
                name = name.replace("_", " ").title()
                
                rows.append(f"""
                <tr>
                    <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:500;">{name}</td>
                    <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right;font-family:monospace;">{latest_val:.4f}</td>
                    <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right;font-family:monospace;color:#666;">{avg_val:.4f}</td>
                    <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right;font-family:monospace;color:#666;">{min_val:.4f} ~ {max_val:.4f}</td>
                    <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right;font-family:monospace;color:#999;">{latest_step}</td>
                </tr>""")
            else:
                rows.append(f"""
                <tr>
                    <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:500;">{tag}</td>
                    <td colspan="4" style="padding:6px 10px;border-bottom:1px solid #eee;color:#999;">No data</td>
                </tr>""")
        except Exception as e:
            rows.append(f"""
            <tr>
                <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:500;">{tag}</td>
                <td colspan="4" style="padding:6px 10px;border-bottom:1px solid #eee;color:#f00;">Error: {e}</td>
            </tr>""")
    
    return "\n".join(rows)


# ============================================================
# 🔊 音频获取
# ============================================================

def download_latest_audio():
    """下载最新的 gen 音频，返回 [(tag, step, wav_path), ...]"""
    audio_files = []
    tags_info = get_audio_tags()
    
    if "eval" not in tags_info:
        print("  ✗ 没有 eval 音频数据")
        return audio_files
    
    gen_tags = [t for t in tags_info["eval"].keys() if t.startswith("gen/")]
    gen_tags.sort()
    
    print(f"  [音频] 获取最新的生成音频 ({len(gen_tags)} 个)...")
    for tag in gen_tags:
        try:
            result = get_latest_audio(tag, "eval")
            if result and result[1]:
                step, wav_data = result
                filename = TMP_DIR / f"audio_{tag.replace('/', '_')}_step{step}.wav"
                with open(filename, 'wb') as f:
                    f.write(wav_data)
                audio_files.append((tag, step, filename))
                print(f"    ✓ {tag} (step {step}, {len(wav_data)} bytes)")
            elif result:
                print(f"    ~ {tag} (step {result[0]}, 无音频数据)")
            else:
                print(f"    ✗ {tag} (无数据)")
        except Exception as e:
            print(f"    ✗ {tag} ({e})")
    
    return audio_files


# ============================================================
# 📧 邮件发送
# ============================================================

def send_email(chart_files, latest_table, error_msg=None):
    """发送邮件报告"""
    global SMTP_CONFIG
    if SMTP_CONFIG is None:
        SMTP_CONFIG = load_email_config()
    cfg = SMTP_CONFIG
    
    # ========== 1. 构建 HTML 正文 ==========
    html_img_tags = []  # 收集 img 标签，稍后插入
    html_parts = []
    
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>So-VITS 训练监控报告</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', Arial, sans-serif; color: #333; margin: 0; padding: 0; background: #f5f5f5; }}
.container {{ max-width: 720px; margin: 0 auto; background: #fff; }}
.header {{ background: linear-gradient(135deg, #1565C0, #0D47A1); color: white; padding: 20px 30px; }}
.header h1 {{ margin: 0; font-size: 22px; }}
.header p {{ margin: 5px 0 0; opacity: 0.9; font-size: 14px; }}
.section {{ margin: 20px; }}
.section h2 {{ font-size: 18px; color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 6px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }}
th {{ background: #f5f5f5; padding: 8px 10px; text-align: right; font-weight: 600; border-bottom: 2px solid #ddd; }}
th:first-child {{ text-align: left; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
td:first-child {{ font-weight: 500; }}
td.num {{ text-align: right; font-family: 'Courier New', monospace; }}
.chart-img {{ width: 100%; max-width: 680px; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; margin: 8px 0; display: block; }}
.footer {{ padding: 15px 20px; background: #fafafa; font-size: 12px; color: #999; text-align: center; border-top: 1px solid #eee; }}
.error {{ background: #ffebee; border: 1px solid #ef5350; padding: 12px 16px; border-radius: 4px; margin: 10px 20px; color: #c62828; }}
.audio-info {{ font-size: 12px; color: #999; margin-top: 8px; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
    <h1>🎵 So-VITS 训练监控报告</h1>
    <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | Step: {get_current_step()}</p>
</div>
""")
    
    if error_msg:
        html_parts.append(f'<div class="error">⚠️ {error_msg}</div>')
    
    # 数值表格
    html_parts.append("""
<div class="section">
<h2>📊 最新指标</h2>
<table>
<thead>
    <tr>
        <th style="text-align:left;">指标</th>
        <th style="text-align:right;">Latest</th>
        <th style="text-align:right;">Avg(近100)</th>
        <th style="text-align:right;">Overall Range</th>
        <th style="text-align:right;">Step</th>
    </tr>
</thead>
<tbody>
""")
    html_parts.append(latest_table)
    html_parts.append("""</tbody>
</table>
</div>
""")
    
    # 图表区域 - 分三组
    group_names = {
        "gradient": "📊 梯度",
        "lr": "🎯 学习率",
        "loss": "📉 损失",
    }
    
    for group_key in ["gradient", "lr", "loss"]:
        group_items = [c for c in chart_files if c[0] == group_key]
        if not group_items:
            continue
        
        html_parts.append(f"""
<div class="section">
<h2>{group_names[group_key]}</h2>
""")
        for group, tag, filepath in group_items:
            if filepath.exists():
                img_cid = f"chart_{tag.replace('/', '_')}"
                html_img_tags.append((img_cid, filepath))
                # 友好的显示名称
                name_map = {
                    "grad_norm_g": "Generator 梯度范数",
                    "grad_norm_d": "Discriminator 梯度范数",
                    "learning_rate": "学习率",
                    "loss/g/total": "Generator 总损失",
                    "loss/g/mel": "Generator Mel 损失",
                    "loss/g/fm": "Generator FM 损失",
                    "loss/g/kl": "Generator KL 损失",
                    "loss/d/total": "Discriminator 总损失",
                }
                display_name = name_map.get(tag, tag)
                html_parts.append(f'<p style="margin:14px 0 4px;font-weight:600;font-size:14px;">{display_name}</p>')
                html_parts.append(f'<img src="cid:{img_cid}" class="chart-img" alt="{display_name}">')
        html_parts.append("</div>")
    
    html_parts.append(f"""
<div class="footer">
<p>So-VITS Training Monitor | <a href="{TENSORBOARD_URL}">{TENSORBOARD_URL}</a></p>
<p>此邮件由 tb_email_monitor.py 自动生成</p>
</div>
</div>
</body>
</html>
""")
    
    html_content = "\n".join(html_parts)
    
    # ========== 2. 构建 MIME 消息 ==========
    # multipart/related: 第一部分必须是根文档(HTML)，后面是嵌入资源(图片)
    msg = MIMEMultipart('related')
    msg['Subject'] = "[So-VITS 训练报告]"
    msg['From'] = cfg['from_addr']
    msg['To'] = ', '.join(cfg['to_addrs'])
    msg['Date'] = formatdate(localtime=True)
    
    # 【关键】HTML 必须是 multipart/related 的第一个部分
    msg.attach(MIMEText(html_content, 'html', 'utf-8'))
    
    # 然后在 HTML 之后附加图片（通过 cid: 引用）
    for img_cid, filepath in html_img_tags:
        with open(filepath, 'rb') as f:
            img_data = f.read()
        img_part = MIMEImage(img_data, _subtype='jpeg')
        img_part.add_header('Content-ID', f'<{img_cid}>')
        img_part.add_header('Content-Disposition', 'inline')
        # 不设置 filename 参数，避免邮件客户端误识别为附件
        msg.attach(img_part)
    
    # ========== 3. 发送 ==========
    print(f"\n  📧 发送邮件到 {', '.join(cfg['to_addrs'])}...")
    try:
        if cfg.get('use_ssl', True):
            server = smtplib.SMTP_SSL(cfg['host'], cfg['port'], timeout=30)
        else:
            server = smtplib.SMTP(cfg['host'], cfg['port'], timeout=30)
            server.starttls()
        
        server.login(cfg['username'], cfg['password'])
        server.sendmail(cfg['from_addr'], cfg['to_addrs'], msg.as_string())
        server.quit()
        print("  ✓ 邮件发送成功")
        return True
    except smtplib.SMTPAuthenticationError:
        print("  ✗ SMTP 认证失败，请检查用户名和授权码")
        print("    提示: 126/163邮箱需要使用「授权码」而非登录密码")
        print("    请在邮箱设置 → POP3/SMTP 中开启服务并获取授权码")
        return False
    except smtplib.SMTPException as e:
        print(f"  ✗ SMTP 错误: {e}")
        print(f"    提示: 如果持续失败，尝试将 use_ssl 改为 {'False' if cfg.get('use_ssl', True) else 'True'}")
        return False
    except Exception as e:
        print(f"  ✗ 发送邮件失败: {e}")
        return False


def get_current_step():
    """获取当前训练步数"""
    try:
        data = get_scalar_data("loss/g/total", ".")
        if data:
            return int(data[-1][0])
    except:
        pass
    return "?"


# ============================================================
# 🚀 主函数
# ============================================================

def send_report():
    """执行一次完整的报告发送"""
    print("=" * 60)
    print(f"  TensorBoard 训练监控报告")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TensorBoard: {TENSORBOARD_URL}")
    print("=" * 60)
    
    error_msg = None
    
    # 1. 检查 TensorBoard 连接
    print("\n[1/4] 检查 TensorBoard 连接...")
    try:
        test = tb_request("/data/plugin/scalars/tags")
        print("  ✓ TensorBoard 连接正常")
    except ConnectionError as e:
        print(f"  ✗ {e}")
        error_msg = f"❌ 无法连接到 TensorBoard ({TENSORBOARD_URL})，请确认训练是否在运行。"
        # 即使连接失败，也尝试发送错误邮件
    except Exception as e:
        print(f"  ✗ 未知错误: {e}")
        error_msg = f"❌ 连接 TensorBoard 时发生未知错误: {e}"
    
    if error_msg:
        # 发送错误通知邮件
        latest_table = "<tr><td colspan='5' style='color:#f00;text-align:center;'>无法获取数据</td></tr>"
        send_email([], [], latest_table, error_msg)
        return
    
    try:
        # 2. 获取标量数据并生成图表
        print("\n[2/4] 生成训练曲线图表...")
        chart_files = generate_all_charts()
        
        # 3. 获取最新数值表格
        print("\n[3/4] 生成最新指标表格...")
        latest_table = generate_latest_values_table()
        
        # 4. 发送邮件
        print("\n[4/4] 发送邮件...")
        send_email(chart_files, latest_table)
        
        # 清理临时文件
        print("\n[清理] 删除临时文件...")
        for _, _, fp in chart_files:
            try:
                fp.unlink(missing_ok=True)
            except:
                pass
        
        print("\n" + "=" * 60)
        print("  ✅ 报告完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n  ❌ 生成报告时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="TensorBoard 训练监控邮件脚本")
    parser.add_argument("--interval", type=int, default=0,
                        help="发送间隔（秒），0 表示只发送一次")
    parser.add_argument("--once", action="store_true",
                        help="发送一次后退出（与 --interval 配合使用时有效）")
    parser.add_argument("--config", type=str, default=None,
                        help="邮件配置文件路径（默认: configs/email_config.json）")
    args = parser.parse_args()
    
    # 加载邮件配置
    global SMTP_CONFIG, CONFIG_FILE
    if args.config:
        # 支持相对/绝对路径，相对路径以当前工作目录为基准
        CONFIG_FILE = Path(args.config).expanduser().resolve()
    print(f"  📧 邮件配置: {CONFIG_FILE}")
    SMTP_CONFIG = load_email_config()
    
    if args.interval > 0:
        print(f"🕐 监控模式：每 {args.interval} 秒发送一次报告")
        count = 0
        while True:
            count += 1
            print(f"\n{'#' * 60}")
            print(f"  # 第 {count} 次报告")
            print(f"{'#' * 60}")
            try:
                send_report()
            except Exception as e:
                print(f"  ❌ 发送报告时崩溃: {e}")
                import traceback
                traceback.print_exc()
            
            if args.once:
                print("\n  --once 模式，退出。")
                break
            
            print(f"\n⏳ 等待 {args.interval} 秒后下一次报告...")
            print(f"   按 Ctrl+C 退出\n")
            time.sleep(args.interval)
    else:
        send_report()


if __name__ == "__main__":
    print("=" * 60)
    print("  TensorBoard 训练监控邮件脚本")
    print("=" * 60)
    print()
    print(f"  📧 邮件配置: {CONFIG_FILE}")
    print(f"  📋 参考模板: {EXAMPLE_FILE}")
    print(f"  🔒 配置已加入 .gitignore，可安全存放密码")
    print()
    
    main()
