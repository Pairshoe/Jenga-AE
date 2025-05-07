import matplotlib.pyplot as plt
import numpy as np
import re

def parse_timing_log(log_path):
    data = []
    current_case = None
    label_map = {"Baseline": "Origin", "Ours": "Jenga"}
    pattern = re.compile(r"(\d+)[kK]: .*?total time:\s*([0-9.]+)")

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line in label_map:
                current_case = label_map[line]
            else:
                match = pattern.search(line)
                if match and current_case:
                    batch_size = match.group(1)
                    total_time = float(match.group(2))
                    data.append({
                        "case": f"{batch_size}K {current_case}",
                        "total_time": total_time
                    })
    return data

data = parse_timing_log("logs/extension/offload/offload.log")
# data = [
#     {"case": "4K Origin",   "total_time": 0},
# ]

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
cases = [d['case'] for d in data]
case_types = ['Origin', 'Jenga']
sequence_lengths = ['4K', '6K', '8K', '10K', '12K', '14K', '16K']

grouped_data = {seq_len: {ctype: 0 for ctype in case_types} for seq_len in sequence_lengths}
for d in data:
    seq_len, ctype = d['case'].split(' ')[0], d['case'].split(' ')[-1]
    grouped_data[seq_len][ctype] = d['total_time']

# calculate the speedup
speedup_data = {seq_len: 0 for seq_len in sequence_lengths}
for seq_len in sequence_lengths:
    speedup_data[seq_len] = grouped_data[seq_len]['Origin'] / grouped_data[seq_len]['Jenga']
print(speedup_data)

# normalize the data (Origin as the baseline)
for seq_len in sequence_lengths:
    for ctype in case_types:
        if ctype != 'Origin':
            grouped_data[seq_len][ctype] /= grouped_data[seq_len]['Origin']
    grouped_data[seq_len]['Origin'] = 1.0

x = np.arange(len(sequence_lengths))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 2))

for i, ctype in enumerate(case_types):
    times = [grouped_data[seq_len][ctype] for seq_len in sequence_lengths]
    ax.bar(x + i * width, times, width, label=ctype, color=colors[i % len(colors) + 3], edgecolor='black', zorder=3)

ax.set_xticks(x + width)
ax.set_xticklabels(sequence_lengths)
ax.set_ylim(0.5, 1.2)
ax.set_xlabel('Sequence Length', fontsize=14)
ax.set_ylabel('Execution Time (ms)', fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('output_figures/extension/offload/exp-extension-offload.pdf')
plt.close()
