#include "bang_unary.h"
namespace infini {

void unary_kernel(cnnlHandle_t handle,
                  const float *input,
                  float *output,
                  const uint32_t num,
                  const uint32_t op_num,
                  UnaryOpType list[]) {
    // 任务类型和调度方法
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    cnrtQueue_t queue;
    cnnlGetQueue(handle, &queue);
    k_dim.x = 4;
    k_dim.y = 8;
    k_dim.z = 1;
    k_type = CNRT_FUNC_TYPE_UNION1;
    uint32_t op_list = 0;
    for(int i = op_num-1; i >= 0; --i) {
      op_list *= 10;
      op_list += list[i];
    }
    // launch 任务
    MLUUnaryKernelUnion1<<<k_dim, k_type, queue>>>((float*)output,
                                                   (float*)input,
                                                   num,
                                                   op_list);
}

};
