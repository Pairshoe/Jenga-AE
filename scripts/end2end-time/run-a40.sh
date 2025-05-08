#!/bin/bash

device="a40"
seq_len=4096

# --------------------------
# LLaMA 模型（llama2 / llama3）
# --------------------------
models_llama=("llama2" "llama3")
methods_llama=("base" "llora" "jenga")

for model in "${models_llama[@]}"; do
  for method in "${methods_llama[@]}"; do
    if [ "$method" == "llora" ]; then
      bash scripts/end2end-time/llama-${method}.sh $model $seq_len True $device
    else
      bash scripts/end2end-time/llama-${method}.sh $model $seq_len True $device
    fi
  done
done

# --------------------------
# OPT 模型
# --------------------------
models_opt=("opt-6.7b" "opt-2.7b" "opt-1.3b" "opt-350m")
methods_opt=("base" "llora" "jenga")

for model in "${models_opt[@]}"; do
  for method in "${methods_opt[@]}"; do
    bash scripts/end2end-time/opt-${method}.sh $model $seq_len True $device
  done
done

# --------------------------
# 画图
# --------------------------
mkdir -p output_figures/end2end/time
python src/experiment/end2end/time/plot_comparison_a40.py
