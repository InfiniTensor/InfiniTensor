rm ./eval_transpose
make -j && ./test_transpose
nvcc ../eval_pfusion/eval_transpose.cu ../generated_code/transpose.cu -I ../eval_pfusion -o eval_transpose
./eval_transpose
