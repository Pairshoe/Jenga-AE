bash scripts/ablation-time-breakdown/llama-jenga.sh llama2  8192
bash scripts/ablation-time-breakdown/llama-jenga.sh llama2  10240
bash scripts/ablation-time-breakdown/llama-jenga.sh llama2  12288
bash scripts/ablation-time-breakdown/llama-jenga.sh llama2  14336
bash scripts/ablation-time-breakdown/llama-jenga.sh llama2  16384
bash scripts/ablation-time-breakdown/llama-baseline.sh llama2  8192
bash scripts/ablation-time-breakdown/llama-baseline.sh llama2  10240

bash scripts/ablation-time-breakdown/predictor.sh

mkdir -p output_figures/ablations/time-breakdown

python src/experiment/ablation/time-breakdown/plot.py