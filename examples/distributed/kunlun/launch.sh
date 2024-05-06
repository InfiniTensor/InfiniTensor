export HF_ENDPOINT=https://hf-mirror.com

# models=("bert" "gpt2" "llama")
models=("bert" "gpt2")
batch_size=(1 32)
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
            # run pytorch model
            echo "Run pytorch $model with batch_size=$bs length=$len ."
            python run_pytorch.py --model "$model" --batch_size "$bs" --length "$len" #> results/"$model"_"$bs"_"$len"_pytorch
            for n in "${nproc[@]}"; do
                # run infinitensor 
                echo "Run $n parallel infinitensor "$model" with batch_size=$bs and length=$len ."
                python kunlun_launch.py --name "$model" --model ../models/"$model"/"$model"_"$bs"_"$len".onnx --nproc_per_node=$n # >> results/"$model"_"$bs"_"$len"_infini 
                # delete internal files
                find ./ -type f -name "*.onnx" -delete
                find ./ -type f -name "*.pb" -delete
            done
            find ./ -type f -name "*.npy" -delete
        done
    done
done
