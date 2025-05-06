import matplotlib.pyplot as plt

data = [
    {"case": "opt350m-lora-4K", "memory": 0},
]

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
length_order = ["4K", "8K", "16K", "32K", "64K"]
methods = ["lora", "longlora", "jenga"]
memory_values = {length: {method: 0 for method in methods} for length in length_order}

for entry in data:
    case_split = entry["case"].split('-')
    method = case_split[1]
    length = case_split[-1]
    if length in memory_values and method in methods:
        memory_values[length][method] = max(memory_values[length][method], entry["memory"])

x_labels = length_order
bar_width = 0.25
x = range(len(x_labels))

plt.figure(figsize=(8, 2))
for i, method in enumerate(methods):
    plt.bar(
        [pos + i * bar_width for pos in x],
        [memory_values[length][method] / 1000 for length in x_labels],
        bar_width,
        label=method.capitalize(),
        color=colors[i % len(colors) + 3],
        edgecolor="black",
        zorder=3,
    )

for i, length in enumerate(x_labels):
    jenga_memory = memory_values[length]["jenga"]
    longlora_memory = memory_values[length]["longlora"]
    if longlora_memory > 0:  # 确保 longlora 内存不为零
        savings = longlora_memory / jenga_memory
        print(f"{length}: {savings:.2f}x")

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.yticks(fontsize=14)
plt.xticks([pos + bar_width for pos in x], x_labels, fontsize=14)
# plt.ylabel("Memory Footprint (GB)", fontsize=14)
plt.tight_layout()
plt.savefig("exp-end2end-memory-opt350m-sequence.pdf")
plt.close()
