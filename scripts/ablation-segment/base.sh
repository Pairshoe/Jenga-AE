#!/bin/bash

# 允许从外部传入参数，否则使用默认值
model=${1:-"llama3"}  # 默认值为 opt-2.7b
max_length=${2:-14336}  # 默认值为 16384
device=${3:-"a800"}  

mkdir -p logs/ablations/memory-breakdown
if [[ "${PYTORCH_CUDA_ALLOC_CONF}" != *"expandable_segments:True"* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
python src/experiment/ablation/segment/base.py\
    --model_name_or_path "checkpoints/${model}" \
    --predictor_path checkpoints/predictor \
    --output_dir ./output/${model}_${max_length} \
    --max_steps 2400 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type constant_with_warmup \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --bf16 \
    --model_max_length "${max_length}" \
    --flash_attention True \
    --pool_size 64 \
    --thresh 0.4 \
