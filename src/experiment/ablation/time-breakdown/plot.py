import matplotlib.pyplot as plt
import os
import re

# data = [
#     {"case": "8K LoRA",   "total_time": 0.0, "forward": 0, "backward": 0, "optimizer step": 0, "prediction": 0},
# ]
def read_from_log():
    log_dir = "logs/ablations/time-breakdown"
    log_files = [
        "llama2-8192-a800-baseline.log",
        "llama2-8192-a800.log",
        "llama2-10000-a800-baseline.log",
        "llama2-10240-a800.log",
        "llama2-12288-a800.log",
        "llama2-14336-a800.log",
        "llama2-16384-a800.log",
    ]

    data = []
    predictor_path = os.path.join(log_dir, "predictor.log")
    predictor_times = {}
    if os.path.exists(predictor_path):
        with open(predictor_path, "r") as f:
            for line in f:
                match = re.search(r"seq_len:\s*(\d+)\s*time:\s*([\d.]+)", line)
                if match:
                    seq_len = int(match.group(1))
                    time = float(match.group(2))
                    predictor_times[seq_len] = time

    for filename in log_files:
        path = os.path.join(log_dir, filename)
        if not os.path.exists(path):
            continue
        
        with open(path, 'r') as f:
            lines = f.readlines()

        # 找最后一行包含"forward time"的
        time_line = None
        for line in reversed(lines):
            if "forward time" in line:
                time_line = line.strip()
                break

        if not time_line:
            continue

        # 提取数值
        match = re.search(
            r"forward time: ([\d.]+), backward time: ([\d.]+), optimizer step time: ([\d.]+), total time: ([\d.]+)",
            time_line
        )
        if not match:
            continue

        forward, backward, optimizer_step, total = map(float, match.groups())

        # 判断 case 名
        base_name = filename.replace(".log", "")
        parts = base_name.split("-")
        size = int(parts[1]) // 1024  # 转换成 K
        tag = "LoRA" if "baseline" in filename else "Jenga"
        case = f"{size}K {tag}"
        # Get predictor time
        prediction_time = 0
        if tag == "Jenga":
            prediction_time = predictor_times.get(size*1024, 0)

        data.append({
            "case": case,
            "total_time": total,
            "forward": forward,
            "backward": backward,
            "optimizer step": optimizer_step,
            "prediction": prediction_time
        })
    print(data)
    return data
data = read_from_log()


for d in data:
    d['total_time'] = d['forward'] + d['backward'] + d['optimizer step'] + d['prediction']
data = data[::-1]

cases = [d['case'] for d in data]
colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
stages = ['forward', 'backward', 'optimizer step', 'prediction']
stage_colors = {stage: colors[i] for i, stage in enumerate(stages)}

fig, ax = plt.subplots(figsize=(8, 3))

bar_height = 0.1
bar_y_based = 0.5
bar_y_gap = 0.1

for i, d in enumerate(data):
    left = 0
    y_pos = i * (bar_height + bar_y_gap) + bar_y_based
    for stage in stages:  
        ax.barh(y_pos, d[stage], left=left, label=stage if i == 0 else "", color=stage_colors[stage], height=bar_height, edgecolor='black')
        left += d[stage]
   
ax.set_xlabel('Execution Time (ms)', fontsize=14)
ax.set_yticks([i * (bar_height + bar_y_gap) + bar_y_based for i in range(len(data))])
ax.set_yticklabels(cases, fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.set_xlim(0, 1.2 * max([d['total_time'] for d in data]))

plt.tight_layout()
plt.savefig('output_figures/ablations/time-breakdown/exp-ablation-time-breakdown.pdf', bbox_inches='tight')


