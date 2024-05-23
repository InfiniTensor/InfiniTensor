 export HF_ENDPOINT=https://hf-mirror.com

models=("bert" "gpt2" "llama")
batch_size=(1 32)
seq_len=(100 500)
nproc=(1 2 4)

for model in "${models[@]}"; do
    for bs in "${batch_size[@]}"; do
        for len in "${seq_len[@]}"; do
            python run_pytorch.py --model "$model" --batch_size "$bs" --length "$len" --export_onnx ../models/"$model" --export_only 
        done
    done
done 
