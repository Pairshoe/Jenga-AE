
mkdir -p output_figures/ablations/predictor
python src/experiment/ablation/predictor/plot_loss.py
python src/experiment/ablation/predictor/elastic_size.py > logs/ablations/predictor/elastic_size.log

mkdir -p output_figures/ablations/algorithm
python src/experiment/ablation/algorithm/plot_llama2_attn.py
python src/experiment/ablation/algorithm/plot_llama2_mlp.py
python src/experiment/ablation/algorithm/plot_opt_attn.py
python src/experiment/ablation/algorithm/plot_opt_mlp.py

mkdir -p output_figures/ablations/memory-breakdown
python src/experiment/ablation/memory-breakdown/plot.py

bash scripts/ablation-segment/base.sh
bash scripts/ablation-segment/segment.sh

mkdir -p output_figures/ablations/time-breakdown
python src/experiment/ablation/time-breakdown/plot.py

mkdir -p output_figures/end2end/memory
python src/experiment/end2end/memory/plot_comparison_4k.py
python src/experiment/end2end/memory/plot_comparison_8k.py
python src/experiment/end2end/memory/plot_sequence_opt1.3b.py
python src/experiment/end2end/memory/plot_sequence_opt350m.py

mkdir -p output_figures/end2end/time
python src/experiment/end2end/time/plot_comparison_a800.py
python src/experiment/end2end/time/plot_sequence.py

mkdir -p output_figures/extension/2d
python src/experiment/extention/2D/plot.py

mkdir -p output_figures/extension/offload
python src/experiment/extention/offload/plot.py