export HF_ENDPOINT=https://hf-mirror.com

# models=("bert" "gpt2" "llama")
models=("llama")
batch_size=(1 )
seq_len=(100 500)
nproc=(1 2 4)

results_dir="results"

if [ -d "$results_dir" ]; then
    echo "directory ./$results_dir exists"
else
    mkdir -p "$results_dir"
    echo "mkdir $results_dir, logs saved there"
fi


for model in "${models[@]}"; do
    for bs in "${batch_size[@]}"; do
        for len in "${seq_len[@]}"; do
            echo "Run pytorch llama with batch_size="$bs" and length="$len""
            python run_pytorch.py --model "$model" --batch_size "$bs" --length "$len"
            for n in "${nproc[@]}"; do
                    # run pytorch model
                    echo "Run infinitensor llama with batch_size="$bs" and length="$len" and nproc="$n"."
                    python kunlun_launch.py --name llama --model ../models/llama/llama_"$bs"_"$len"_fp32.onnx --nproc_per_node=$n
                    # delete internal files
                    find ./ -type f -name "*.onnx" -delete
                    find ./ -type f -name "*0c" -delete
            done
            find ./ -type f -name "*.npy" -delete
        done
    done
done
