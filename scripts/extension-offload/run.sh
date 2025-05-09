mkdir -p output_figures/extension/offload
mkdir -p logs/extension/offload

for seqlen in 4096 6144 8192 10240 12288 14336 16384
do
    bash scripts/extension-offload/base.sh llama2  $seqlen 
done

for seqlen in 4096 6144 8192 10240 12288 14336 16384
do
    bash scripts/extension-offload/base.sh llama2  $seqlen 
done



# python src/experiment/extention/offload/plot.py