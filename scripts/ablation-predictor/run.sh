bash scripts/ablaion-predictor/llama_rp.sh
bash scripts/ablaion-predictor/opt_rp.sh

bash scripts/ablaion-predictor/llama_la.sh
bash scripts/ablaion-predictor/opt_la.sh


mkdir -p logs/ablations/predictor
mv output/llama2_16384_predictor_la logs/ablations/predictor/
mv output/llama2_16384_predictor_red logs/ablations/predictor/
mv output/opt-6.7b_16384_predictor_la logs/ablations/predictor/
mv output/opt-6.7b_16384_predictor_red logs/ablations/predictor/

mkdir -p output_figures/ablations/predictor
python src/experiment/ablation/predictor/plot_loss.py

python src/experiment/ablation/predictor/elastic_size.py > logs/ablations/predictor/elastic_size.log