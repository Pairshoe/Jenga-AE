mkdir -p logs/end2end/accuracy


rm -rf logs/end2end/accuracy/ppl-jenga-pp.log
rm -rf logs/end2end/accuracy/ppl-lora-pp.log

for seq_len in 8192 10240 12288 14336 16384
do

    python src/experiment/end2end/accuracy/ppl.py --seq_len $seq_len \
        --peft_model checkpoints/peft_model/rp/$((seq_len/1024))k/jenga \
        --data_path dataset/PPL/test_pg19.bin \
        >> logs/end2end/accuracy/ppl-jenga-pp.log

    python src/experiment/end2end/accuracy/ppl.py --seq_len $seq_len \
        --peft_model checkpoints/peft_model/rp/$((seq_len/1024))k/lora \
        --data_path dataset/PPL/test_pg19.bin \
        >> logs/end2end/accuracy/ppl-lora-pp.log

done