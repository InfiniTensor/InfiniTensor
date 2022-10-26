rm ./eval_sar_drn_0 ./eval_sar_drn_1 ./eval_sar_drn_2
make -j && ./test_sar_drn
nvcc ../eval_pfusion/eval_sar_drn_0.cu ../generated_code/sar_drn_0.cu -I ../eval_pfusion -o eval_sar_drn_0
nvcc ../eval_pfusion/eval_sar_drn_1.cu ../generated_code/sar_drn_0.cu -I ../eval_pfusion -o eval_sar_drn_1
nvcc ../eval_pfusion/eval_sar_drn_2.cu ../generated_code/sar_drn_1.cu -I ../eval_pfusion -o eval_sar_drn_2
./eval_sar_drn_0
./eval_sar_drn_1
./eval_sar_drn_2
