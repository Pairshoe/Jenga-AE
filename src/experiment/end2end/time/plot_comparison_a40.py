import matplotlib.pyplot as plt

data = [
    {"case": "llama2-lora-a40", "time": 0},
]

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']
models = ["llama3", "llama2", "opt6.7b", "opt2.7b", "opt1.3b", "opt350m"]
methods = ["lora", "longlora", "jenga"]

x_labels = []
time_values = {method: [] for method in methods}
speedups_labels = []

for model in models:
    x_labels.append(model)
    jenga_time = 0
    longlora_time = 0
    for method in methods:
        case = f"{model}-{method}-a40"
        time = next((d["time"] for d in data if d["case"] == case), 0)
        time_values[method].append(time)
        if method == "jenga":
            jenga_time = time
        if method == "longlora":
            longlora_time = time
    if longlora_time > 0:
        speedups = longlora_time / jenga_time
        speedups_labels.append(f"{speedups:.2f}x")
    else:
        speedups_labels.append("N/A")

print(speedups_labels)

for i in range(len(models)):
    if time_values["longlora"][i] == 0:
        time_values["jenga"][i] = 1
        continue
    time_values["lora"][i] /= time_values["longlora"][i]
    time_values["jenga"][i] /= time_values["longlora"][i]
    time_values["longlora"][i] = 1

bar_width = 0.25
x = range(len(models))

plt.figure(figsize=(8, 2))
for i, method in enumerate(methods):
    plt.bar(
        [pos + i * bar_width for pos in x],
        time_values[method],
        bar_width,
        label=method.capitalize(),
        color=colors[i % len(colors)],
        edgecolor="black",
        zorder=3,
    )

plt.ylim(0.5, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.yticks(fontsize=14)
plt.xticks([pos + bar_width for pos in x], x_labels, fontsize=14)
# plt.ylabel("Execution Time (ms)", fontsize=14)
plt.tight_layout()
plt.savefig("output_figures/end2end/time/exp-end2end-time-a40-comparison.pdf")
plt.close()
