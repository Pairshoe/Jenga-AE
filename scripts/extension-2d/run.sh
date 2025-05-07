bash scripts/extension-2d/llama-jenga.sh llama2 16384
bash scripts/extension-2d/llama-2d.sh llama2 16384
bash scripts/extension-2d/llama-base.sh llama2 16384

bash scripts/extension-2d/llama-jenga.sh llama2 32768
bash scripts/extension-2d/llama-2d.sh llama2 32768
bash scripts/extension-2d/llama-base.sh llama2 32768


bash scripts/extension-2d/llama-jenga.sh llama2 49152
bash scripts/extension-2d/llama-2d.sh llama2 49152
bash scripts/extension-2d/llama-base.sh llama2 49152

bash scripts/extension-2d/llama-jenga.sh llama2 65536
bash scripts/extension-2d/llama-2d.sh llama2 65536
bash scripts/extension-2d/llama-base.sh llama2 65536

mkdir -p output_figures/extension/2d
python src/experiment/extention/2D/plot.py

