#include "bang/bang_common.h"

__global__ void bangPrintFloatImpl(float *x, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%.3f ", x[i]);
    }
    printf("\n");
}

namespace infini {

void bangPrintFloat(cnrtQueue_t queue, float *x, int len) {

   cnrtDim3_t k_dim;
   cnrtFunctionType_t k_type;
   k_dim.x = 1;
   k_dim.y = 1;
   k_dim.z = 1;
   k_type = CNRT_FUNC_TYPE_BLOCK;

    bangPrintFloatImpl<<<k_dim, k_type, queue>>>(x, len);
    cnrtSyncDevice();
}

} // namespace infini
