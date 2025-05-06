#!/bin/bash

lengths=(8192 10240 12288 14336 16384)
mkdir -p logs/ablations/time-breakdown
rm  logs/ablations/time-breakdown/predictor.log
for length in "${lengths[@]}"; do
    echo "Running predictor time breakdown for sequence length: ${length}"
    python src/experiment/ablation/time-breakdown/test_predictor_time.py \
        --seq_len "${length}" \
        >> "logs/ablations/time-breakdown/predictor.log"
    # Add your processing commands here
done