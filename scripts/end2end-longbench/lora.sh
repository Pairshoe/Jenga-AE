mkdir -p logs/end2end/accuracy/longbench

python src/experiment/end2end/accuracy/longbench.py --peft_model checkpoints/peft_model/la/lora
python src/experiment/end2end/accuracy/eval.py --model lora