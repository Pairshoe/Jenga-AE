# 4K
bash scripts/end2end-time/llama-base.sh llama3 8192
bash scripts/end2end-time/llama-llora.sh llama3 8192
bash scripts/end2end-time/llama-jenga.sh llama3 8192
bash scripts/end2end-time/llama-base.sh llama2 8192
bash scripts/end2end-time/llama-llora.sh llama2 8192
bash scripts/end2end-time/llama-jenga.sh llama2 8192

bash scripts/end2end-time/opt-base.sh opt-6.7b 8192
bash scripts/end2end-time/opt-llora.sh opt-6.7b 8192
bash scripts/end2end-time/opt-jenga.sh opt-6.7b 8192

bash scripts/end2end-time/opt-base.sh opt-2.7b 8192
bash scripts/end2end-time/opt-llora.sh opt-2.7b 8192
bash scripts/end2end-time/opt-jenga.sh opt-2.7b 8192

bash scripts/end2end-time/opt-base.sh opt-1.3b 8192
bash scripts/end2end-time/opt-llora.sh opt-1.3b 8192
bash scripts/end2end-time/opt-jenga.sh opt-1.3b 8192

bash scripts/end2end-time/opt-base.sh opt-350m 8192
bash scripts/end2end-time/opt-llora.sh opt-350m 8192
bash scripts/end2end-time/opt-jenga.sh opt-350m 8192

# seq
bash scripts/end2end-time/llama-base.sh llama2 4096 True
bash scripts/end2end-time/llama-jenga.sh llama2 4096 True
bash scripts/end2end-time/llama-base.sh llama3 4096 True
bash scripts/end2end-time/llama-jenga.sh llama3 4096 True

bash scripts/end2end-time/llama-base.sh llama2 8192 True
bash scripts/end2end-time/llama-jenga.sh llama2 8192 True
bash scripts/end2end-time/llama-base.sh llama3 8192 True
bash scripts/end2end-time/llama-jenga.sh llama3 8192 True

bash scripts/end2end-time/llama-base.sh llama2 16384 True
bash scripts/end2end-time/llama-jenga.sh llama2 16384 True
bash scripts/end2end-time/llama-base.sh llama3 16384 True
bash scripts/end2end-time/llama-jenga.sh llama3 16384 True

bash scripts/end2end-time/llama-base.sh llama2 32768 True
bash scripts/end2end-time/llama-jenga.sh llama2 32768 True
bash scripts/end2end-time/llama-base.sh llama3 32768 True
bash scripts/end2end-time/llama-jenga.sh llama3 32768 True

bash scripts/end2end-time/llama-base.sh llama2 49152 True
bash scripts/end2end-time/llama-jenga.sh llama2 49152 True
bash scripts/end2end-time/llama-base.sh llama3 49152 True
bash scripts/end2end-time/llama-jenga.sh llama3 49152 True

bash scripts/end2end-time/llama-base.sh llama2 65536 True
bash scripts/end2end-time/llama-jenga.sh llama2 65536 True
bash scripts/end2end-time/llama-base.sh llama3 65536 True
bash scripts/end2end-time/llama-jenga.sh llama3 65536 True





mkdir -p output_figures/end2end/time

python src/experiment/end2end/time/plot_comparison_a800.py
python src/experiment/end2end/time/plot_sequence.py
