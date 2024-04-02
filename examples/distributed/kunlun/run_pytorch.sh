export HF_ENDPOINT=https://hf-mirror.com

models=("bert" "gpt2")
batch_size=(1 32)
seq_len=(100 500)
nproc=(1 2 4)

for model in "${models[@]}"; do
    for bs in "${batch_size[@]}"; do
        for len in "${seq_len[@]}"; do
            python -m xacc run_pytorch.py --model "$model" --batch_size "$bs" --length "$len" --export_onnx ../models/"$model" > results/"$model"_"$bs"_"$len" 
                for n in "${nproc[@]}"; do
                    python kunlun_launch.py --name "$model" --model ../models/"$model"/"$model"_"$bs"_"$len".onnx --nproc_per_node=$n >> results/"$model"_"$bs"_"$len" 
                done
        done
    done
done
