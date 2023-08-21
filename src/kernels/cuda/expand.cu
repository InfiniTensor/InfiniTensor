//#include <iostream>
//#include <cuda_runtime.h>
#include "cuda/cuda_common.h"

//#define BLOCK_DIM ((int)2)

__global__ 
void _expand_kernel(float *d_input, float *d_output, int inputsize, int outputsize){//d_input,d_ooutput都是1D向量
    
    int i = threadIdx.x + blockIdx.x*blockDim.x; 
    
    if(i < outputsize){
        d_output[i] = d_input[i%inputsize];
    }
}

namespace infini {
void expand_kernel(float *d_input, float *d_output, int inputsize, int outputsize) {
    int blocksize = 32 * 16;
    int gridsize = (outputsize + blocksize - 1) / blocksize;
    _expand_kernel<<<blocksize, gridsize>>>(d_input, d_output, inputsize, outputsize);
}
} // namespace infini
/***
//下面涉及**d_vec这种二维指针的函数全是有问题的
void cudaExpand1D(const float* inputData, float* outputData, int inputsize, int outputsize) {
    float* devInputData;
    float* devOutputData;

    cudaMalloc((void**)&devInputData, inputsize * sizeof(float));
    cudaMalloc((void**)&devOutputData, outputsize * sizeof(float));

    cudaMemcpy(devInputData, inputData, inputsize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(outputsize/(float)(BLOCK_DIM)),1,1);
    dim3 block_dim(BLOCK_DIM, 1, 1);

    _expand_kernel <<<grid_dim, block_dim>>> (devInputData, devOutputData, inputsize, outputsize);

    cudaMemcpy(outputData, devOutputData, outputsize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devInputData);
    cudaFree(devOutputData);
}
__global__
void flattenkernel(float **d_input2d, float *d_input1d, int row, int col){//d_input1d,d_input_2d分别是gpu上的1D,2D数组
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(i < row && j < col){
        d_input1d[j*row + i] = d_input2d[i][j];
    }

}
void cudaExpandflatten(float** input2d, int row, int col) {
    int inputsize = row*col;
    float** d_input2d = (float**)malloc(row * sizeof(float*));//GPU上的2D数组，用来接收CPU上inpu2d的数据
    
    for (int i = 0; i < row; i++) {
        cudaMalloc((void**)&d_input2d[i], col * sizeof(float));
        cudaMemcpy(d_input2d[i], input2d[i], col * sizeof(float), cudaMemcpyHostToDevice);
    }
    float *d_input1d;//GPU上的1D数组，用来接收flatten以后的gpu上的2D数组信息
    cudaMalloc((void **) &d_input1d, inputsize*sizeof(float));
    dim3 grid_dim(ceil(row/(float)(BLOCK_DIM)),ceil(col/(float)(BLOCK_DIM)),1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    flattenkernel<<<grid_dim, block_dim>>>(d_input2d, d_input1d, row, col);
    float *h_input1d = (float *)malloc(inputsize*sizeof(float));//CPU上的1D数组,检验flatten的gpu1D数组数据是否正确
    cudaMemcpy(h_input1d,d_input1d, inputsize*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < inputsize; i++){
        std::cout << h_input1d[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < row; i++) {
        cudaFree(d_input2d[i]);
    }
    cudaFree(d_input2d);
    cudaFree(d_input1d);
    free(h_input1d);

}

__global__ 
void expand2D(float **d_input, float **d_output, int row, int col, int outrow, int outcol){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y; 
    if(i < outrow && j < outcol){
        //d_output[i][j] = d_input[i%row][j%col];
        d_output[i][j] = 3 + i + j;
    }
}

void cudaExpand2D(float** inputData, float** outputData, int row, int col, int outrow, int outcol) {
    
    float** devInputData = (float**)malloc(row * sizeof(float*));
    float** devOutputData = (float**)malloc(outrow * sizeof(float*));
    //cudaMalloc((void**)&devInputData, row * sizeof(float*));
    //cudaMalloc((void**)&devOutputData, outrow * sizeof(float*));
    
    for (int i = 0; i < row; i++) {
        cudaMalloc((void**)&devInputData[i], col * sizeof(float));
        cudaMemcpy(devInputData[i], inputData[i], col * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(inputData[i],devInputData[i],  col * sizeof(float), cudaMemcpyDeviceToHost);
        
    }
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            std::cout << inputData[i][j] << " ";
        }
        std::cout << " \n";
    }
    for (int i = 0; i < outrow; i++) {
        cudaMalloc((void**)&devOutputData[i], outcol * sizeof(float));
    }

    dim3 grid_dim(ceil(outrow/(float)(BLOCK_DIM)),ceil(outcol/(float)(BLOCK_DIM)),1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    expand2D<<<grid_dim, block_dim>>>(devInputData, devOutputData, row, col, outrow, outcol);
    for (int i = 0; i < outrow; i++) {
        cudaMemcpy(outputData[i], devOutputData[i], outcol * sizeof(float), cudaMemcpyDeviceToHost);
    }
    for(int i = 0; i < outrow; i++){
        for(int j = 0; j < outcol; j++){
            std::cout << outputData[i][j] << " ";
        }
    }

    for (int i = 0; i < row; i++) {
        cudaFree(devInputData[i]);
    }
    cudaFree(devInputData);
    for(int i = 0; i < outrow; i++){
        cudaFree(devOutputData[i]);
    }
    cudaFree(devOutputData);
}
int main() {
    const int inputsize = 3;
    const int factor = 2;
    int outputsize = (factor == -1 ? inputsize:factor*inputsize);
    float inputData[inputsize] = {1, 2, 3};
    float *outputData;
    outputData = (float *)malloc(factor*inputsize*sizeof(float));

    cudaExpand1D(inputData, outputData, inputsize, outputsize);

    for (int i = 0; i < outputsize; i++) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;
    //-----------------
    int row = 1;
    int col = 3;
    int fac2d[2] = {1,1};
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
    float **out2d;
    int outrow = row*fac2d[0];
    int outcol = col*fac2d[1];
    out2d = (float**)malloc(outrow * sizeof(float*));
    for (int i = 0; i < outrow; i++) {
        out2d[i] = (float*)malloc(outcol * sizeof(float));
    }
    cudaExpand2D(in2d, out2d, row, col, outrow, outcol);
    for(int i = 0; i < outrow; i++){
        for(int j = 0; j < outcol; j++){
            std::cout << out2d[i][j] << " ";
        }
        std::cout << " \n";
    }
    //--------------
    cudaExpandflatten(in2d, row, col);
    for (int i = 0; i < row; i++) {
        free(in2d[i]);
    }
    free(in2d);
    for (int i = 0; i < outrow; i++) {
        free(out2d[i]);
    }
    free(out2d);
    return 0;
}
***/
