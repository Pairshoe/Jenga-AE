mkdir -p logs/end2end/accuracy


rm -rf logs/end2end/accuracy/ppl-jenga.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 8192 \
    >> logs/end2end/accuracy/ppl-jenga.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 10240 \
    >> logs/end2end/accuracy/ppl-jenga.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 12288 \
    >> logs/end2end/accuracy/ppl-jenga.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 14336 \
    >> logs/end2end/accuracy/ppl-jenga.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 16384 \
    >> logs/end2end/accuracy/ppl-jenga.log

rm -rf logs/end2end/accuracy/ppl-baseline.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 8192 \
    --baseline \
    >> logs/end2end/accuracy/ppl-baseline.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 10240 \
    --baseline \
    >> logs/end2end/accuracy/ppl-baseline.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 12288 \
    --baseline \
    >> logs/end2end/accuracy/ppl-baseline.log

python src/experiment/end2end/accuracy/ppl.py --seq_len 14336 \
    --baseline \
    >> logs/end2end/accuracy/ppl-baseline.log
    
python src/experiment/end2end/accuracy/ppl.py --seq_len 16384 \
    --baseline \
    >> logs/end2end/accuracy/ppl-baseline.log