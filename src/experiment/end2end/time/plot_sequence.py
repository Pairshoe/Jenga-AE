import matplotlib.pyplot as plt
import numpy as np
import os
import re

# data_llama2 = {
#     "Seq": ["4k", "8k", "16k", "32k", "48k", "64k"],
#     "lora": [1241.929, 2666.448, 6047.240, 15439.142, 28292.347, 44677.158],
#     "ours": [1149.568, 2390.249, 5202.09, 12336.110, 21521.058, 32813.567],
# }
# data_llama3 = {
#     "Seq": ["4k", "8k", "16k", "32k", "48k", "64k"],
#     "lora": [1310.354, 2788.882, 6327.096, 15967.367, 0.0, 0.0],
#     "ours": [1253.038, 2593.830, 5572.286, 13049.171, 22555.839, 34640.386],
# }

def parse_llama_logs(log_dir):
    data_llama2 = {"Seq": ["4k", "8k", "16k", "32k", "48k", "64k"], "lora": [], "ours": []}
    data_llama3 = {"Seq": ["4k", "8k", "16k", "32k", "48k", "64k"], "lora": [], "ours": []}
    seq_map = {"4096": "4k", "8192": "8k", "16384": "16k", "32768": "32k", "49152": "48k", "65536": "64k"}

    for filename in os.listdir(log_dir):
        if not filename.startswith("checkpoint-") or not filename.endswith(".log") or "llora" in filename or 'a800' not in filename:
            continue
        parts = filename.split("-")
        if "llama2" in filename:
            model_data = data_llama2
        elif "llama3" in filename:
            model_data = data_llama3
        else:
            continue
        seq = [p for p in parts if p.isdigit()]
        if not seq or seq[0] not in seq_map:
            continue
        seq_str = seq_map[seq[0]]
        filepath = os.path.join(log_dir, filename)
        
        if "baseline" in filename:
            method = "lora"
        elif "llora" in filename:
            method = "longlora"
        else:
            method = "ours"
            
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            match = re.search(r"total time:\s*([\d.]+)", lines[-1])
            if not match:
                model_data[method].append(0.0)
                continue
            total_time = float(match.group(1))
            # Place in correct sequence position
            idx = model_data["Seq"].index(seq_str)
            while len(model_data["lora"]) <= idx:
                model_data["lora"].append(0.0)
            while len(model_data["ours"]) <= idx:
                model_data["ours"].append(0.0)

            model_data[method][idx] = round(total_time, 3)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return data_llama2, data_llama3

data_llama2, data_llama3 = parse_llama_logs("logs/end2end/time")

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

seq_lengths = data_llama2["Seq"]
bar_width = 0.2
x = np.arange(len(seq_lengths))

# calculate the speedup
speedup_llama2 = [data_llama2["lora"][i] / data_llama2["ours"][i] for i in range(len(seq_lengths))]
speedup_llama3 = [data_llama3["lora"][i] / data_llama3["ours"][i] if data_llama3["lora"][i] is not None else None for i in range(len(seq_lengths))]
print(speedup_llama2)
print(speedup_llama3)

# normalize the data with the largest value in one sequence length
for length in seq_lengths:
    idx = seq_lengths.index(length)
    max_val = max(data_llama2["lora"][idx], data_llama2["ours"][idx], data_llama3["lora"][idx], data_llama3["ours"][idx])
    data_llama2["lora"][idx] /= max_val
    data_llama2["ours"][idx] /= max_val
    data_llama3["lora"][idx] /= max_val
    data_llama3["ours"][idx] /= max_val

fig, ax = plt.subplots(figsize=(8, 2))

plt.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

# Llama2
ax.bar(x - bar_width * 1.5, data_llama2["lora"], bar_width, color=colors[0], edgecolor="black", zorder=3)
ax.bar(x - bar_width * 0.5, data_llama2["ours"], bar_width, color=colors[1], edgecolor="black", zorder=3)

# Llama3
ax.bar(x + bar_width * 0.5, [v if v is not None else 0 for v in data_llama3["lora"]], bar_width, color=colors[2], edgecolor="black", zorder=3)
ax.bar(x + bar_width * 1.5, [v if v is not None else 0 for v in data_llama3["ours"]], bar_width, color=colors[3], edgecolor="black", zorder=3)

plt.ylim(0.5, 1.1)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(seq_lengths)
ax.set_xlabel("Sequence Length", fontsize=14)

plt.tight_layout()
plt.savefig("output_figures/end2end/time/exp-end2end-time-sequence.pdf")
