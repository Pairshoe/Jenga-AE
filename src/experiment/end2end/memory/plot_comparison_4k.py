import matplotlib.pyplot as plt
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
                lines = [line.strip() for line in f if "reserve" in line]
        except Exception as e:
            continue  # Skip unreadable files

        # Determine case name
        base = filename.replace(".log", "")
        if "baseline" in base:
            case_type = "lora"
            base = base.replace("baseline", "")
        elif "llora" in base:
            case_type = "longlora"
            base = base.replace("llora", "")
        else:
            case_type = "jenga"
        
        # Extract context length and model
        parts = base.split("-")
        if "opt" not in filename:
            model = parts[0]
            context = parts[1]  # remove the trailing K if present
        else:
            model = parts[0]
            size = parts[1]
            model = model+size
            context = parts[2]
        context = int(context) // 1024
        case_name = f"{model}-{case_type}-{context}K"

        # Check if OOM
        if len(lines) < 10:
            memory = 0
        else:
            # Extract last 'reserve' value
            match = re.search(r"reserve:\s*([\d.]+)", lines[-1])
            memory = float(match.group(1)) if match else 0

        data.append({"case": case_name, "memory": memory})
    print(data)
    return data


# data = [
#     {"case": "llama2-lora-4K", "memory": 0},
# ]

data = parse_memory_logs("logs/end2end/memory")

data_4k = [d for d in data if "4K" in d["case"]]

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
models = ["llama3", "llama2", "opt6.7b", "opt2.7b", "opt1.3b", "opt350m"]
methods = ["lora", "longlora", "jenga"]

x_labels = []
memory_values = {method: [] for method in methods}
savings_labels = []

for model in models:
    x_labels.append(model)
    jenga_memory = 0
    longlora_memory = 0
    for method in methods:
        case = f"{model}-{method}-4K"
        memory = next((d["memory"] / 1000 for d in data_4k if d["case"] == case), 0)
        memory_values[method].append(memory)
        if method == "jenga":
            jenga_memory = memory
        if method == "longlora":
            longlora_memory = memory
    if longlora_memory > 0:
        savings = longlora_memory / jenga_memory
        savings_labels.append(f"{savings:.2f}x")
    else:
        savings_labels.append("N/A")

print(savings_labels)

for i in range(len(models)):
    for method in methods:
        if method != "longlora":
            memory_values[method][i] /= memory_values["longlora"][i]
    memory_values["longlora"][i] = 1

bar_width = 0.25
x = range(len(models))

plt.figure(figsize=(8, 2))
for i, method in enumerate(methods):
    plt.bar(
        [pos + i * bar_width for pos in x],
        memory_values[method],
        bar_width,
        label=method.capitalize(),
        color=colors[i % len(colors)],
        edgecolor="black",
        zorder=3,
    )

# for i, label in enumerate(savings_labels):
#     jenga_x_pos = x[i] + 2 * bar_width
#     jenga_y_pos = memory_values["jenga"][i]
#     plt.text(jenga_x_pos, jenga_y_pos + 100, label, ha="center", va="bottom", fontsize=10)

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.yticks(fontsize=14)
plt.xticks([pos + bar_width for pos in x], x_labels, fontsize=14)
# plt.ylabel("Memory Footprint (GB)", fontsize=14)
plt.tight_layout()
plt.savefig("output_figures/end2end/memory/exp-end2end-memory-4K-comparison.pdf")
plt.close()
