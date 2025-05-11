models=("llama2" "opt-6.7b")
seq_lens=(1024 2048 4096)
cars_number=(1 2 4)

for model in "${models[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        for num_gpu in "${cars_number[@]}"; do
            echo "Running $model $seq_len $num_gpu"
            bash scripts/scalability/time.sh $model $seq_len $num_gpu
        done
    done
done

mkdir -p output_figures/scalability

python src/experiment/scalability/plot_llama2.py
python src/experiment/scalability/plot_opt.py

