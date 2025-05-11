import matplotlib.pyplot as plt
import os
import re

def parse_memory_logs(log_dir):
    data = []

    for filename in os.listdir(log_dir):
        if not filename.endswith(".log") or "opt" not in filename:
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if "total time" in line]
        except Exception as e:
            continue  # Skip unreadable files
        
        # Extract context length and model
        filename = filename.replace(".log", "")
        parts = filename.split("_")

        model = parts[0]
        seq_len = parts[1]
        cards_number = parts[2]

            
        case_name = f"{cards_number}-{int(seq_len)//1024}K"


        # Extract last 'reserve' value
        if len(lines) < 3:
            time = 0.0
        else:
            match = re.search(r"total time:\s*([\d.]+)", lines[-1])
            time = float(match.group(1)) if match else 0

        data.append({"case": case_name, "total_time": time/int(cards_number)})
    print(data)
    return data

data = parse_memory_logs("logs/scalability")
# data = [
#     {"case": "1-1K", "total_time": 745.257},
#     {"case": "2-1K", "total_time": 989.985},
#     {"case": "4-1K", "total_time": 1733.753},
#     {"case": "1-2K", "total_time": 923.087},
#     {"case": "2-2K", "total_time": 1342.000},
#     {"case": "4-2K", "total_time": 2023.740},
#     {"case": "1-4K", "total_time": 1707.879},
#     {"case": "2-4K", "total_time": 2149.883},
#     {"case": "4-4K", "total_time": 2456.912},
# ]


colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
case_types = ['1', '2', '4']
sequence_lengths = ['1K', '2K', '4K']

grouped_data = {seq_len: [] for seq_len in sequence_lengths}
for d in data:
    case_type, seq_len = d['case'].split('-')
    grouped_data[seq_len].append((case_type, d['total_time']))

for seq_len in grouped_data:
    grouped_data[seq_len].sort(key=lambda x: int(x[0]))

plt.figure(figsize=(4, 3))
for i, seq_len in enumerate(sequence_lengths):
    card_numbers = [int(card[0]) for card in grouped_data[seq_len]]  # 卡数
    times = [card[1] for card in grouped_data[seq_len]]  # 时间
    plt.plot(card_numbers, times, marker='o', label=f"Seq {seq_len}", color=colors[i])

plt.xlabel("GPU Number", fontsize=14)
plt.yticks(fontsize=14)
plt.xticks([1, 2, 4], fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
plt.tight_layout()

plt.savefig("./output_figures/scalability/exp-scalability-opt.pdf")
plt.close()
