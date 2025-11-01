import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 基本配置
# ===============================
LOG_ROOT = "logs"       # 根目录：包含 exp1, exp2, exp3, exp4
OUT_DIR = "result"      # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

EXPS = {
    "EXP1: d=128,h=4,L=2,ep=10": "exp1",
    "EXP2: d=256,h=8,L=3,ep=10": "exp2",   # 主实验：重点突出
    "EXP3: d=512,h=8,L=6,ep=10": "exp3",
    "EXP4: d=256,h=8,L=3,ep=20": "exp4",
}

# 各实验对应的颜色与标记样式
STYLES = {
    "EXP1": ("#1f77b4", "o"),   # 蓝
    "EXP2": ("#d62728", "s"),   # 红（主实验）
    "EXP3": ("#2ca02c", "D"),   # 绿
    "EXP4": ("#ff7f0e", "^"),   # 橙
}

CSV_PATTERNS = ["train_log.csv", "*.csv"]
SMOOTH_WINDOW = 1  # 平滑窗口（可改为 3、5 试试更平滑）

# ===============================
# 工具函数
# ===============================
def read_csv_files(exp_dir):
    dfs = []
    for pat in CSV_PATTERNS:
        for file in glob.glob(os.path.join(exp_dir, pat)):
            try:
                df = pd.read_csv(file)
                if 'epoch' not in df.columns:
                    df['epoch'] = np.arange(1, len(df) + 1)
                dfs.append(df)
            except Exception as e:
                print(f"❌ 读取失败: {file}", e)
    return dfs

def aggregate(dfs, metric):
    if not dfs:
        return None, None, None
    max_epoch = max(df['epoch'].max() for df in dfs)
    all_runs = []
    for df in dfs:
        ser = pd.Series(index=np.arange(1, max_epoch+1), dtype=float)
        ser.loc[df['epoch']] = df[metric].values
        ser = ser.fillna(method='ffill')
        all_runs.append(ser.values)
    mat = np.vstack(all_runs)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    epochs = np.arange(1, len(mean)+1)
    return epochs, mean, std

def smooth(y, window):
    if window <= 1:
        return y
    return np.convolve(y, np.ones(window)/window, mode='same')

# ===============================
# 绘图函数
# ===============================
def plot_metric(metric_name, ylabel, out_name, highlight_label="EXP2"):
    plt.figure(figsize=(10,6))
    for label, subdir in EXPS.items():
        exp_dir = os.path.join(LOG_ROOT, subdir)
        dfs = read_csv_files(exp_dir)
        epochs, mean, std = aggregate(dfs, metric_name)
        if epochs is None:
            print(f"⚠️ 无数据: {label}")
            continue

        mean = smooth(mean, SMOOTH_WINDOW)
        color, marker = STYLES.get(label.split(":")[0], ("gray", "o"))

        # 重点实验更显眼
        lw = 3 if highlight_label in label else 2
        z = 10 if highlight_label in label else 1

        plt.plot(epochs, mean, label=label, color=color, marker=marker, lw=lw, zorder=z)
        plt.fill_between(epochs, mean-std, mean+std, color=color, alpha=0.1)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs Epoch", fontsize=14)
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(save_path, dpi=200)
    print(f"✅ 保存图像: {save_path}")
    plt.close()

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    print("🚀 开始绘图...")
    plot_metric("val_loss", "Validation Loss", "combined_val_loss.png")
    plot_metric("bleu", "BLEU Score", "combined_bleu.png")
    print("🎉 所有图像已保存到:", OUT_DIR)
