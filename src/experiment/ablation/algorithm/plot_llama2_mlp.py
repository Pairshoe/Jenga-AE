import matplotlib.pyplot as plt

import re

def parse_log_to_data(log_path):
    data = []
    pattern = re.compile(r"layer:\s*(\d+)\s*,\s*threshold:\s*([0-9.]+)\s*,\s*memory:\s*([0-9.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                layer = int(match.group(1))
                threshold = float(match.group(2))
                memory = float(match.group(3))
                data.append({
                    "layer": layer,
                    "threshold": threshold,
                    "memory": memory
                })
    return data

data = parse_log_to_data("logs/ablations/algorithm/llama2-mlp.log")

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

layers = [d["layer"] for d in data]
thresholds = [d["threshold"] for d in data]
memories = [d["memory"] for d in data]

# average
threshold_avg = sum(thresholds) / len(thresholds)
memory_avg = sum(memories) / len(memories)
print(f"threshold_avg: {threshold_avg}")
print(f"memory_avg: {memory_avg}")

fig, ax1 = plt.subplots(figsize=(3, 2))
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# memory
ax1.plot(layers, memories, color='black', marker='o', markersize=3, markerfacecolor=colors[2], markeredgewidth=0.5, zorder=100, linewidth=1)
ax1.set_ylim(0, 1)
ax1.set_xticks(range(1, 33, 5))

# threshold
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.plot(layers, thresholds, color='black', marker='o', markersize=3, markerfacecolor=colors[3], markeredgewidth=0.5, zorder=100, linewidth=1)

plt.tight_layout()
plt.savefig("output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-mlp.pdf")
plt.close()
