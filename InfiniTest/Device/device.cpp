#include <pybind11/pybind11.h>

#ifdef BANG
#include "cnrt.h"
#endif

#ifdef CUDA
#include "cuda_runtime.h"
#endif

namespace py = pybind11;

int add(int i , int j) {
    return i + j;
}

#ifdef BANG
cnrtNotifier_t bang_start, bang_end;
cnrtQueue_t bang_queue;

bool BangCreateQueue() {
    cnrtRet_t ret = cnrtSetDevice(0);
    if (ret != cnrtSuccess) {
        return false;
    }
    ret = cnrtQueueCreate(&bang_queue);
    if (ret != cnrtSuccess) {
        return false;
    }
    return true;
}

void* BangGetQueue() {
    return bang_queue;
}

bool BangPlaceStartNotifier(void* queue) {
    cnrtQueue_t queue_ptr = reinterpret_cast<cnrtQueue_t>(queue);
    cnrtRet_t ret = cnrtNotifierCreate(&bang_start);
    if (ret != cnrtSuccess) {
        return false;
    }
    ret = cnrtPlaceNotifier(bang_start, queue_ptr);
    if (ret != cnrtSuccess) {
        return false;
    }
    return true;
}

bool BangPlaceEndNotifier(void* queue) {
    cnrtQueue_t queue_ptr = reinterpret_cast<cnrtQueue_t>(queue);
    cnrtRet_t ret = cnrtNotifierCreate(&bang_end);
    if (ret != cnrtSuccess) {
        return false;
    }
    ret = cnrtPlaceNotifier(bang_end, queue_ptr);
    if (ret != cnrtSuccess) {
        return false;
    }
    return true;
}

float BangGetNotifierDuration(void* queue) {
    float time;
    cnrtQueue_t queue_ptr = reinterpret_cast<cnrtQueue_t>(queue);
    cnrtRet_t ret = cnrtQueueSync(queue_ptr);
    if (ret != cnrtSuccess) {
        return false;
    }
    cnrtQueryNotifier(bang_start);
    cnrtQueryNotifier(bang_end);
    cnrtWaitNotifier(bang_start);
    cnrtWaitNotifier(bang_end);
    cnrtNotifierDuration(bang_start, bang_end, &time);
    cnrtNotifierDestroy(bang_start);
    cnrtNotifierDestroy(bang_end);
    return time;
}
#endif

#ifdef CUDA
cudaEvent_t cuda_start, cuda_end;
cudaStream_t cuda_queue;

bool CudaCreateQueue() {
    cudaError_t ret = cudaSetDevice(0);
    if (ret != cudaSuccess) {
        return false;
    }
    ret = cudaStreamCreate(&cuda_queue);
    if (ret != cudaSuccess) {
        return false;
    }
    return true;
}

void* CudaGetQueue() {
    return cuda_queue;
}

bool CudaPlaceStartNotifier(void* queue) {
    cudaStream_t queue_ptr = reinterpret_cast<cudaStream_t>(queue);
    cudaError_t ret = cudaEventCreate(&cuda_start);
    if (ret != cudaSuccess) {
        return false;
    }
    ret = cudaEventRecord(cuda_start, queue_ptr);
    if (ret != cudaSuccess) {
        return false;
    }
    return true;
}

bool CudaPlaceEndNotifier(void* queue) {
    cudaStream_t queue_ptr = reinterpret_cast<cudaStream_t>(queue);
    cudaError_t ret = cudaEventCreate(&cuda_end);
    if (ret != cudaSuccess) {
        return false;
    }
    ret = cudaEventRecord(cuda_end, queue_ptr);
    if (ret != cudaSuccess) {
        return false;
    }
    return true;
}

float CudaGetNotifierDuration(void* queue) {
    float time;
    cudaStream_t queue_ptr = reinterpret_cast<cudaStream_t>(queue);
    cudaError_t ret = cudaEventSynchronize(cuda_end);
    if (ret != cudaSuccess) {
        return false;
    }
    cudaEventQuery(cuda_start);
    cudaEventQuery(cuda_end);
    cudaStreamWaitEvent(queue_ptr, cuda_start, 0);
    cudaStreamWaitEvent(queue_ptr, cuda_end, 0);
    cudaEventElapsedTime(&time, cuda_start, cuda_end);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_end);
    return time;
}
#endif

PYBIND11_MODULE(device, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, pybind11::arg("i"), pybind11::arg("j"), "a func that adds two numbers");
#ifdef BANG
    m.def("bangCreateQueue", &BangCreateQueue, "a func create queue");
    m.def("bangGetQueue", &BangGetQueue, "a func get queue");
    m.def("bangPlaceStartNotifier", &BangPlaceStartNotifier, "a func place the start notifier");
    m.def("bangPlaceEndNotifier", &BangPlaceEndNotifier, "a func place the end notifier");
    m.def("bangGetNotifierDuration", &BangGetNotifierDuration, "a func get the hardware time from start notifier to end notifier");
#endif

#ifdef CUDA
    m.def("cudaCreateQueue", &CudaCreateQueue, "a func create queue");
    m.def("cudaGetQueue", &CudaGetQueue, "a func get queue");
    m.def("cudaPlaceStartNotifier", &CudaPlaceStartNotifier, "a func place the start notifier");
    m.def("cudaPlaceEndNotifier", &CudaPlaceEndNotifier, "a func place the end notifier");
    m.def("cudaGetNotifierDuration", &CudaGetNotifierDuration, "a func get the hardware time from start notifier to end notifier");
#endif
}


