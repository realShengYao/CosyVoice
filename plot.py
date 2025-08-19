import os
import pandas as pd
import matplotlib.pyplot as plt

def collect_data(root_path):
    results = []
    # 遍历所有子文件夹
    for subdir, _, files in os.walk(root_path):
        if "data.csv" in files:
            file_path = os.path.join(subdir, "data.csv")
            df = pd.read_csv(file_path)
            # 计算平均值
            hardware = df["hardware"].iloc[0]
            avg_latency = df["first_token"].mean()
            avg_rtf = df["rtf"].mean()
            results.append([hardware, avg_latency, avg_rtf])
    return pd.DataFrame(results, columns=["hardware", "avg_latency", "avg_rtf"])

def plot_results(df, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Latency 图
    plt.figure(figsize=(8, 6))
    plt.bar(df["hardware"], df["avg_latency"])
    plt.ylabel("Average First Token Latency (s)")
    plt.xlabel("Hardware")
    plt.title("Hardware vs First Token Latency")
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"), dpi=300)
    plt.close()

    # RTF 图
    plt.figure(figsize=(8, 6))
    plt.bar(df["hardware"], df["avg_rtf"])
    plt.ylabel("Average RTF")
    plt.xlabel("Hardware")
    plt.title("Hardware vs RTF")
    plt.savefig(os.path.join(output_dir, "rtf_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    root_path = "benchmark/"  # <-- 修改为你的置顶路径
    df = collect_data(root_path)
    print(df)  # 打印结果表格
    plot_results(df)