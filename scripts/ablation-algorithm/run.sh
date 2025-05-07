bash scripts/ablation-algorithm/llama-attn.sh
bash scripts/ablation-algorithm/llama-mlp.sh
bash scripts/ablation-algorithm/opt-attn.sh
bash scripts/ablation-algorithm/opt-mlp.sh

mkdir -p output_figures/ablations/algorithm
python src/experiment/ablation/algorithm/plot_llama2_attn.py
python src/experiment/ablation/algorithm/plot_llama2_mlp.py
python src/experiment/ablation/algorithm/plot_opt_attn.py
python src/experiment/ablation/algorithm/plot_opt_mlp.py