//#include <iostream>
//#include <cuda_runtime.h>
#include "cuda/cuda_common.h"
//#define BLOCK_DIM ((int)2)

__global__
void _where_kernel(const float* condition, const float* input, const float* other, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = condition[index] ? input[index] : other[index];
    }
}


namespace infini {
void where_kernel(const float* condition, const float* input, const float* other, float* output, int size) {
    int blocksize = 32 * 16;
    int gridsize = (size + blocksize - 1) / blocksize;
    _extend_kernel<<<blocksize, gridsize>>>(condition, input, other, output, size);
}
} // namespace infini
/***
//发现cuda似乎不支持kernel函数接受二维指针，因此默认全部是1D数组
//下面涉及**d_vec这种二维指针的函数全是有问题的
void where1D(const float* condition, const float* input, const float* other, float* output, int size) {
    // 计算所需的线程块和线程数量
    dim3 grid_dim(ceil(size/(float)(BLOCK_DIM)),1,1);
    dim3 block_dim(BLOCK_DIM, 1, 1);

    // 在 GPU 上分配内存
    float* d_condition;
    float* d_input;
    float* d_other;
    float* d_output;
    cudaMalloc((void**)&d_condition, size * sizeof(float));
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_other, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // 将数据从主机内存复制到 GPU 上
    cudaMemcpy(d_condition, condition, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_other, other, size * sizeof(float), cudaMemcpyHostToDevice);

    // 调用 GPU 上的核函数
    _where_kernel<<<grid_dim, block_dim>>>(d_condition, d_input, d_other, d_output, size);

    // 将结果从 GPU 复制回主机内存
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 上的内存
    cudaFree(d_condition);
    cudaFree(d_input);
    cudaFree(d_other);
    cudaFree(d_output);
}
__global__
void flattenkernel(float **d_input2d, float *d_input1d, int row, int col){//d_input1d,d_input_2d分别是gpu上的1D,2D数组
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(i < row && j < col){
        d_input1d[i*col + j] = 2*(i + j);
    }

}
void flattencpu(float **input2d, float *input1d, int row, int col){//CPU上的flatten函数
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            input1d[i*col + j] = input2d[i][j];
        }
        
    }

}
void cudaflatten(float** input2d, int row, int col) {
    int inputsize = row*col;
    float** d_input2d = (float**)malloc(row * sizeof(float*));//GPU上的2D数组，用来接收CPU上inpu2d的数据
    float** h2d = (float**)malloc(row * sizeof(float*));
    for(int i = 0; i < row; i++){
        h2d[i] = (float*)malloc(col * sizeof(float));
    }
    for (int i = 0; i < row; i++) {
        cudaMalloc((void**)&d_input2d[i], col * sizeof(float));
        cudaMemcpy(d_input2d[i], input2d[i], col * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h2d[i],d_input2d[i],  col * sizeof(float), cudaMemcpyDeviceToHost);
    }
    for(int i = 0; i< row; i++){
        for(int j = 0; j< col; j++){
            std::cout << h2d[i][j] << " ";
        }
        std::cout << "\n" ;
    }
    float *d_input1d;//GPU上的1D数组，用来接收flatten以后的gpu上的2D数组信息
    cudaMalloc((void **) &d_input1d, inputsize*sizeof(float));
    
    float *h_input1d = (float *)malloc(inputsize*sizeof(float));//CPU上的1D数组,检验flatten的gpu1D数组数据是否正确
    //cudaMemcpy(h_input1d,d_input1d, inputsize*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < inputsize; i++){
        std::cout << h_input1d[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    dim3 grid_dim(ceil(row/(float)(BLOCK_DIM)),ceil(col/(float)(BLOCK_DIM)),1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    flattenkernel<<<grid_dim, block_dim>>>(d_input2d, d_input1d, row, col);
    
    //float *h_input1d = (float *)malloc(inputsize*sizeof(float));//CPU上的1D数组,检验flatten的gpu1D数组数据是否正确
    cudaMemcpy(h_input1d,d_input1d, inputsize*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < inputsize; i++){
        std::cout << h_input1d[i] << " ";
    }
    std::cout << std::endl;
    flattencpu(input2d,h_input1d,row,col);
    for(int i = 0; i < inputsize; i++){
        std::cout << h_input1d[i] << " ";
    }
    for (int i = 0; i < row; i++) {
        cudaFree(d_input2d[i]);
    }
    cudaFree(d_input2d);
    cudaFree(d_input1d);
    free(h_input1d);

}
int main() {
    int size = 10;

    float* condition;
    float* input;
    float* other;
    float* output;
    condition = (float *)malloc(size*sizeof(float));
    input = (float *)malloc(size*sizeof(float));
    other = (float *)malloc(size*sizeof(float));
    output = (float *)malloc(size*sizeof(float));

    for (int i = 0; i < size; i++) {
        condition[i] = i % 2 == 0 ? 1 : 0;
        input[i] = i;
        other[i] = -i;
    }

    where1D(condition, input, other, output, size);

    std::cout << "Output: ";
    for (int i = 0; i < size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    free(condition);
    free(input);
    free(other);
    free(output);
    //-------------------
    int row = 2,col = 3;
    float **in2d = (float **)malloc(row * sizeof(float *));
    for (int i = 0; i < row; i++) {
        in2d[i] = (float *)malloc(col * sizeof(float));
    }

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            in2d[i][j] = i + j;
            std::cout << in2d[i][j] << " ";
        }
        std::cout << " \n";
    }
    cudaflatten(in2d, row, col);
    for (int i = 0; i < row; i++) {
        free(in2d[i]);
    }
    free(in2d);
    return 0;
}
***/
