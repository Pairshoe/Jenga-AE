import matplotlib.pyplot as plt
import os
import re

# data = [
#     {"case": "8K LoRA",     "total_mem": 0.0, "model_states": 0, "activations": 0, "others": 0, "predictors": 0},
# ]
def read_from_log():
    log_dir = "logs/ablations/memory-breakdown"
    log_files = [
        "llama2-8192-a800-baseline.log",
        "llama2-8192-a800-llora.log",
        "llama2-8192-a800.log",
        "llama2-10240-a800.log",
        "llama2-12288-a800.log",
        "llama2-14336-a800.log",
        "llama2-16384-a800.log",
    ]

    data = []

    # 解析 predictor.log
    predictor_path = os.path.join(log_dir, "predictor.log")
    predictor_memories = 0.0
    if os.path.exists(predictor_path):
        with open(predictor_path, "r") as f:
            for line in f:
                match = re.search(r"Total memory:\s*([\d.]+)\s*MB", line)
                if match:
                    memory = float(match.group(1))
                    predictor_memories = memory

    for filename in log_files:
        path = os.path.join(log_dir, filename)
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            lines = [line.strip() for line in f if "allocaiton" in line]

        if len(lines) < 2:
            continue

        # 提取 second allocation (index 1)
        second_alloc = re.search(r"allocaiton:\s*([0-9.]+)", lines[1])
        second_alloc_val = float(second_alloc.group(1)) if second_alloc else 0.0

        # 最后一行的信息
        last_line = lines[-1]
        last_alloc = re.search(r"allocaiton:\s*([0-9.]+)", last_line)
        reserve = re.search(r"reserve:\s*([0-9.]+)", last_line)

        if not last_alloc or not reserve:
            continue

        model_states = float(last_alloc.group(1))
        total_mem = float(reserve.group(1))
        activations = second_alloc_val - model_states
        others = total_mem - second_alloc_val

        # 判断 case 名
        base_name = filename.replace(".log", "")
        parts = base_name.split("-")
        size = int(parts[1]) // 1024  # 转换成 K
        if "baseline" in filename:
            tag = "LoRA"
        elif "llora" in filename:
            tag = "LongLoRA"
        else:
            tag = "Jenga"
        case = f"{size}K {tag}"

        # 获取 predictor memory（只对 Jenga 有效）
        predictor_mem = predictor_memories if tag == "Jenga" else 0.0

        data.append({
            "case": case,
            "total_mem": round(total_mem, 2),
            "model_states": round(model_states, 2),
            "activations": round(activations, 2),
            "others": round(others, 2),
            "predictors": round(predictor_mem, 2)
        })

    print(data)
    return data
data = read_from_log()


for d in data:
    d['model_states'] /= 1024
    d['activations'] /= 1024
    d['others'] /= 1024
    d['predictors'] /= 1024
    d['total_mem'] = d['model_states'] + d['activations'] + d['others'] + d['predictors']
data = data[::-1]

cases = [d['case'] for d in data]
colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
stages = ['model_states', 'activations', 'others', 'predictors']
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
   
ax.set_xlabel('Memory Footprint (GB)', fontsize=14)
ax.set_yticks([i * (bar_height + bar_y_gap) + bar_y_based for i in range(len(data))])
ax.set_yticklabels(cases, fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.set_xlim(0, 1.2 * max([d['total_mem'] for d in data]))

plt.tight_layout()
plt.savefig('output_figures/ablations/memory-breakdown/exp-ablation-mem-breakdown.pdf', bbox_inches='tight')