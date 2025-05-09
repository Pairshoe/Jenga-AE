#!/bin/bash

# --------------------------
# 运行 8192 上的全部模型组合
# --------------------------
models_llama=("llama2" "llama3")
methods_llama=("base" "llora" "jenga")

for model in "${models_llama[@]}"; do
  for method in "${methods_llama[@]}"; do
    bash scripts/end2end-time/llama-${method}.sh $model 8192
  done
done

models_opt=("opt-6.7b" "opt-2.7b" "opt-1.3b" "opt-350m")
methods_opt=("base" "llora" "jenga")

for model in "${models_opt[@]}"; do
  for method in "${methods_opt[@]}"; do
    bash scripts/end2end-time/opt-${method}.sh $model 8192
  done
done

# --------------------------
# LLaMA 不同 seq_len 实验
# --------------------------
seq_lens=(4096 8192 16384 32768 49152 65536)
models_seq=("llama2" "llama3")
methods_seq=("base" "jenga")  # 只有 base 和 jenga

for model in "${models_seq[@]}"; do
  for seq_len in "${seq_lens[@]}"; do
    for method in "${methods_seq[@]}"; do
      bash scripts/end2end-time/llama-${method}.sh $model $seq_len True
    done
  done
done

# --------------------------
# 画图
# --------------------------
mkdir -p output_figures/end2end/time
python src/experiment/end2end/time/plot_comparison_a800.py
python src/experiment/end2end/time/plot_sequence.py
