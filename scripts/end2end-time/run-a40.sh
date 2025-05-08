bash scripts/end2end-time/llama-base.sh llama3 8192 a40
bash scripts/end2end-time/llama-llora.sh llama3 8192 a40
bash scripts/end2end-time/llama-jenga.sh llama3 8192 a40
bash scripts/end2end-time/llama-base.sh llama2 8192 a40
bash scripts/end2end-time/llama-llora.sh llama2 8192 a40
bash scripts/end2end-time/llama-jenga.sh llama2 8192 a40

bash scripts/end2end-time/opt-base.sh opt-6.7b 8192 a40
bash scripts/end2end-time/opt-llora.sh opt-6.7b 8192 a40
bash scripts/end2end-time/opt-jenga.sh opt-6.7b 8192 a40

bash scripts/end2end-time/opt-base.sh opt-2.7b 8192 a40
bash scripts/end2end-time/opt-llora.sh opt-2.7b 8192 a40
bash scripts/end2end-time/opt-jenga.sh opt-2.7b 8192 a40

bash scripts/end2end-time/opt-base.sh opt-1.3b 8192 a40
bash scripts/end2end-time/opt-llora.sh opt-1.3b 8192 a40
bash scripts/end2end-time/opt-jenga.sh opt-1.3b 8192 a40

bash scripts/end2end-time/opt-base.sh opt-350m 8192 a40
bash scripts/end2end-time/opt-llora.sh opt-350m 8192 a40
bash scripts/end2end-time/opt-jenga.sh opt-350m 8192 a40

mkdir -p output_figures/end2end/time

python src/experiment/end2end/time/plot_comparison_a40.py
