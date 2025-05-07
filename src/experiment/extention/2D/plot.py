import matplotlib.pyplot as plt
import numpy as np
import os
import re

def parse_memory_logs(log_dir):
    data = []

    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if "total time" in line]
        except Exception as e:
            continue  # Skip unreadable files

        # Determine case name
        base = filename.replace(".log", "")
        if "baseline" in base:
            case_type = "0D"
            base = base.replace("baseline", "")
        elif "2D" in base:
            case_type = "2D"
            base = base.replace("2D", "")
        else:
            case_type = "1D"
        
        # Extract context length and model
        parts = base.split("-")

        context = parts[1]
        context = int(context) // 1024
        case_name = f"{context}K {case_type}"

        match = re.search(r"total time:\s*([\d.]+)", lines[-1])
        time = float(match.group(1)) if match else 0

        data.append({"case": case_name, "total_time": time})
    print(data)
    return data


# data = [
#     {"case": "16K 0D",  "total_time": 0},
# ]

data = parse_memory_logs("logs/extension/2d")
colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
cases = [d['case'] for d in data]
case_types = ['0D', '1D', '2D']
sequence_lengths = ['16K', '32K', '48K', '64K']

grouped_data = {seq_len: {ctype: 0 for ctype in case_types} for seq_len in sequence_lengths}
for d in data:
    seq_len, ctype = d['case'].split(' ')[0], d['case'].split(' ')[-1]
    grouped_data[seq_len][ctype] = d['total_time']

# calculate the speedup
speedup_data_1D = {seq_len: 0 for seq_len in sequence_lengths}
speedup_data_2D = {seq_len: 0 for seq_len in sequence_lengths}
for seq_len in sequence_lengths:
    speedup_data_1D[seq_len] = grouped_data[seq_len]['0D'] / grouped_data[seq_len]['1D']
    speedup_data_2D[seq_len] = grouped_data[seq_len]['0D'] / grouped_data[seq_len]['2D']
print(speedup_data_1D)
print(speedup_data_2D)

x = np.arange(len(sequence_lengths))
width = 0.25

# normalize the data (0D as the baseline)
for seq_len in sequence_lengths:
    for ctype in case_types:
        if ctype != '0D':
            grouped_data[seq_len][ctype] /= grouped_data[seq_len]['0D']
    grouped_data[seq_len]['0D'] = 1.0

fig, ax = plt.subplots(figsize=(8, 2))

for i, ctype in enumerate(case_types):
    times = [grouped_data[seq_len][ctype] for seq_len in sequence_lengths]
    ax.bar(x + i * width, times, width, label=ctype, color=colors[i % len(colors)], edgecolor='black', zorder=3)

ax.set_xticks(x + width)
ax.set_xticklabels(sequence_lengths)
ax.set_ylim(0.25, 1.2)
ax.set_xlabel('Sequence Length', fontsize=14)
ax.set_ylabel('Total Time (ms)', fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('output_figures/extension/2d/exp-extension-2d.pdf')
plt.close()
