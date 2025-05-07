MAX_LENGTH=16384
MODEL="llama2"
pool_size=64
thresh=0.4

python src/experiment/ablation/predictor/llama_rp.py \
    --model_name_or_path checkpoints/$MODEL \
    --output_dir output/${MODEL}_${MAX_LENGTH}_predictor_red \
    --max_steps 400 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type constant_with_warmup \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --logging_steps 1 \
    --bf16 \
    --model_max_length $MAX_LENGTH \
    --flash_attention True \
    --pool_size ${pool_size} \
    --thresh ${thresh} \
    --save_only_model True \