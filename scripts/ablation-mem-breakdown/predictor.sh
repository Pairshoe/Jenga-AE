mkdir -p logs/ablations/memory-breakdown

python src/experiment/ablation/memory-breakdown/test_predictor_mem.py \
    > "logs/ablations/memory-breakdown/predictor.log"