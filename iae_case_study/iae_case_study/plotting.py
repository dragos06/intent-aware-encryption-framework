import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("../results/cicids_run/summary_results_*.json")
if not files:
    raise FileNotFoundError("No summary JSON file found in results/cicds_run/")
json_path = files[-1]

with open(json_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.fillna(0, inplace=True)

outdir = "results/cicds_run/plots"
os.makedirs(outdir, exist_ok=True)

def save_bar_plot(x, y, ylabel, title, filename):
    plt.figure(figsize=(10,6))
    plt.bar(x, y, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_bar_plot(df['policy'], df['avg_latency_ms'], 'Average Latency (ms)',
              'Average Latency per Policy', f"{outdir}/avg_latency.png")

save_bar_plot(df['policy'], df['avg_energy_units'], 'Average Energy Units',
              'Average Energy per Policy', f"{outdir}/avg_energy.png")

save_bar_plot(df['policy'], df['throughput_ops_per_sec'], 'Throughput (ops/sec)',
              'Throughput per Policy', f"{outdir}/throughput.png")

iae_df = df[df['policy'].str.startswith("IAE")]
if not iae_df.empty:
    save_bar_plot(iae_df['policy'], iae_df['clf_acc'], 'Classifier Accuracy',
                  'Intent Classification Accuracy (IAE Policies)',
                  f"{outdir}/clf_accuracy.png")

print(f"Plots saved to {outdir}")
