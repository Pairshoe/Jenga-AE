#!/bin/bash

# 允许从外部传入参数，否则使用默认值
model=${1:-"llama2"}  # 默认值为 opt-2.7b
max_length=${2:-8192}  # 默认值为 16384
gradient_checkpointing=${3:-"False"}  # 默认值为 False
device=${4:-"a800"}  

mkdir -p logs/end2end/time
# if [[ "${PYTORCH_CUDA_ALLOC_CONF}" != *"expandable_segments:True"* ]]; then
#     export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# fi

# 根据 gradient_checkpointing 的值设置日志文件名
if [[ "${gradient_checkpointing}" == "True" ]]; then
    log_file="logs/end2end/time/checkpoint-${model}-${max_length}-${device}-baseline.log"
else
    log_file="logs/end2end/time/${model}-${max_length}-${device}-baseline.log"
fi


# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python src/experiment/end2end/time/llama_base.py \
    --model_name_or_path "checkpoints/${model}" \
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
    --gradient_checkpoint "${gradient_checkpointing}" \
    > "${log_file}"
