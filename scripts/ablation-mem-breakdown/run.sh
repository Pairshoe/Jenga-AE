bash scripts/ablation-mem-breakdown/llama-llora.sh llama2 8192
bash scripts/ablation-mem-breakdown/llama-base.sh llama2 8192

bash scripts/ablation-mem-breakdown/llama-jenga.sh llama2 8192
bash scripts/ablation-mem-breakdown/llama-jenga.sh llama2 10240
bash scripts/ablation-mem-breakdown/llama-jenga.sh llama2 12288
bash scripts/ablation-mem-breakdown/llama-jenga.sh llama2 14336
bash scripts/ablation-mem-breakdown/llama-jenga.sh llama2 16384

bash scripts/ablation-mem-breakdown/predictor.sh

mkdir -p output_figures/ablations/memory-breakdown
python src/experiment/ablation/memory-breakdown/plot.py


