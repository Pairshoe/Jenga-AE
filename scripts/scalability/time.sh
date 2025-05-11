pool_size=64
thresh=0.4
model=${1:-"llama2"}
max_length=${2:-16384}
num_gpus=${3:-4}



mkdir -p logs/scalability/

if [[ "${PYTORCH_CUDA_ALLOC_CONF}" != *"expandable_segments:True"* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi


deepspeed --num_gpus=${num_gpus}  src/experiment/scalability/time.py \
    --model_name_or_path checkpoints/$model \
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
    --model_max_length $max_length \
    --flash_attention True \
    --pool_size ${pool_size} \
    --thresh ${thresh} \
    --deepspeed src/experiment/scalability/ds_config/stage2.json \
    --gradient_checkpoint True \
    > logs/scalability/${model}_${max_length}_${num_gpus}.log 