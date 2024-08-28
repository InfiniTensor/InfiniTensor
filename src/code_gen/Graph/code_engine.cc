#include "code_gen/code_engine.h"
#include "code_gen/nnet/Visitor/AsTVMVisitor.h"
#include "code_gen/perf_engine.h"
#include "code_gen/transpose.h"
#include "ffi/ffi_embed.h"
#include "fmt/core.h"
#include <fstream>
#include <sys/stat.h>
#include <unordered_set>

namespace tpm {

std::string CodeEngine::actToStr(Operator::ActType act) {
    switch (act) {
    case Operator::None:
        return "CUDNN_ACTIVATION_IDENTITY";
    case Operator::Relu:
        return "CUDNN_ACTIVATION_RELU";
    case Operator::Sigmoid:
        return "CUDNN_ACTIVATION_SIGMOID";
    case Operator::Tanh:
        return "CUDNN_ACTIVATION_TANH";
    default:
        assert(false);
    }
}

std::pair<std::string, std::string> CodeEngine::genTranspose(
    const std::vector<const TransposeOp *> ops, const std::string &funcName,
    const std::string &inputName, const std::string &outputName) {
    std::vector<std::shared_ptr<TransBasic>> microOps;
    assert(!ops.empty());
    for (const TransposeOp *op : ops)
        for (auto &&item : op->getTTParam())
            microOps.emplace_back(item);

    auto &&dimA = getDim(*ops.front()->getInputs()[0]);
    auto &&dimO = getDim(*ops.back()->getOutputs()[0]);

    std::vector<TransBasic *> microOps_;
    microOps_.reserve(microOps.size());
    for (auto &&item : microOps)
        microOps_.emplace_back(item.get());
    auto lambda = TransposeEngine::getInstance().getLambda(microOps_, dimA);

    auto size = lambda.size();
    if (lambda.substr(size - 11, 11) == "I[0][a,c,b]") {
        assert(dimA.size() == 3);
        std::cout << "Transpose 3d: a,c,b hit!" << std::endl;
        std::string invoke_code = "";
        invoke_code += "{\n";
        invoke_code += "    dim3 gridSize(80, 1);\n";
        invoke_code += "    dim3 blockSize(32 * 4, 1);\n";
        invoke_code += "    kernel_transpose_3d<<<gridSize, blockSize>>>(" +
                       inputName + ", ";
        invoke_code += outputName + ", ";
        invoke_code += std::to_string(dimA[0]) + ", ";
        invoke_code += std::to_string(dimA[1]) + ", ";
        invoke_code += std::to_string(dimA[2]) + ");\n";
        invoke_code += "    cudaCheckError();\n";
        invoke_code += "}\n";
        return std::make_pair("", invoke_code);
    }
    if (lambda.substr(size - 13, 13) == "I[0][a,d,b,c]") {
        assert(dimA.size() == 4);
        std::cout << "Transpose 4d: a,d,b,c hit!" << std::endl;
        std::string invoke_code = "";
        invoke_code += "{\n";
        invoke_code += "    dim3 gridSize(80, 1);\n";
        invoke_code += "    dim3 blockSize(32 * 4, 1);\n";
        invoke_code += "    kernel_transpose_3d<<<gridSize, blockSize>>>(" +
                       inputName + ", ";
        invoke_code += outputName + ", ";
        invoke_code += std::to_string(dimA[0]) + ", ";
        invoke_code += std::to_string(dimA[1] * dimA[2]) + ", ";
        invoke_code += std::to_string(dimA[3]) + ");\n";
        invoke_code += "    cudaCheckError();\n";
        invoke_code += "}\n";
        return std::make_pair("", invoke_code);
    }

    auto ret = getTVMCode({dimA}, {"float32"}, dimO, lambda, funcName,
                          {inputName}, outputName);
    return ret;
}

inline bool CodeEngine::check_existed(const std::string &filename) {
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}

std::string CodeEngine::genCode(std::shared_ptr<SubGraph> &graph) {
    clear();
    genHeader();
    std::string line;
    line = "int main() {";
    emit(line);
    shiftTab(1);
    emit("cudnnHandle_t cudnn;");
    emit("cublasHandle_t cublas;");
    emit("checkCudnnError(cudnnCreate(&cudnn));");
    emit("checkCublasError(cublasCreate(&cublas));");

    emit("size_t wsSize = 7ll << 30;");
    emit("float *wsData;");
    emit("checkCudaError(cudaMalloc((void **) &wsData, wsSize));");

    emit("curandGenerator_t gen;");
    emit("checkCurandError(curandCreateGenerator(&gen, "
         "CURAND_RNG_PSEUDO_DEFAULT));");
    emit("checkCurandError(curandSetPseudoRandomGeneratorSeed(gen, (unsigned "
         "long long)clock()));");

    auto tensors = graph->getTensors();
    for (auto t : tensors) {
        genTensorAlloc(*t);
    }
    // FIXME: Hack for bias
    for (auto &&op : graph->getOperators()) {
        if (op->getType() == Operator::Conv) {
            Tensor *t = ((ConvOp *)op)->getBias();
            if (t != nullptr) {
                genTensorAlloc(*t, true);
            }
        }
        if (op->getType() == Operator::Matmul) {
            Tensor *t = ((MatmulOp *)op)->getBias();
            if (t != nullptr) {
                genTensorAlloc(*t);
            }
        }
    }

    // Optimization for Split and Concat
    for (auto &&op : graph->getOperators()) {
        if (op->getType() == Operator::Split &&
            ((SplitOp *)op)->getDim() == 0) {
            int offset = 0;
            auto &&in = op->getInputs()[0];
            for (auto &&out : op->getOutputs()) {
                emit(fmt::format("{} = {} + {};", getVarName(*out),
                                 getVarName(*in), std::to_string(offset)));
                offset += getTensorNElem(*out);
            }
        }
        if (op->getType() == Operator::Concat &&
            ((ConcatOp *)op)->getDim() == 0) {
            int offset = 0;
            auto &&out = op->getOutput();
            for (auto &&in : op->getInputs()) {
                emit(fmt::format("{} = {} + {};", getVarName(*in),
                                 getVarName(*out), std::to_string(offset)));
                offset += getTensorNElem(*in);
            }
        }
    }

    // reversed DFS post-order is topo-order
    std::unordered_set<const Operator *> flag;
    std::vector<Operator *> opsRev;
    std::function<void(Operator *)> dfs = [&](Operator *op) {
        if (flag.count(op)) {
            return;
        }
        flag.insert(op);
        for (auto &&next : op->getSuccessors()) {
            dfs(next);
        }
        opsRev.emplace_back(op);
    };
    for (auto &&op : graph->getOperators()) {
        dfs(op);
    }

    emit("");
    emit("// Desc");
    for (auto it = opsRev.rbegin(); it != opsRev.rend(); it++) {
        genDesc(**it);
    }

    emit("cudaEvent_t st, ed;");
    emit("float duration;");
    emit("checkCudaError(cudaEventCreate(&st));");
    emit("checkCudaError(cudaEventCreate(&ed));");

    emit("");
    emit("// Compute");
    emit("// Preprocess operators for weights");
    for (auto it = opsRev.rbegin(); it != opsRev.rend(); it++) {
        if ((*it)->getInputs().size() == 1 &&
            (*it)->getInputs()[0]->getType() == Tensor::Weight) {
            genCompute(**it);
        }
    }
    emit("// Count time for other operators");
    emit("int warmup = 100, rounds = 1000;");
    emit("for (int i = 0; i < warmup + rounds; i++) {");
    shiftTab(1);
    emit("if (i == warmup) {");
    shiftTab(1);
    emit("checkCudaError(cudaEventRecord(st, 0));");
    shiftTab(-1);
    emit("}");
    for (auto it = opsRev.rbegin(); it != opsRev.rend(); it++) {
        if ((*it)->getInputs().size() > 1 ||
            (*it)->getInputs()[0]->getType() != Tensor::Weight) {
            genCompute(**it);
        }
    }
    shiftTab(-1);
    emit("}");
    assert(transposeMap.empty());

    emit("checkCudaError(cudaEventRecord(ed, 0));");
    emit("checkCudaError(cudaEventSynchronize(st));");
    emit("checkCudaError(cudaEventSynchronize(ed));");
    emit("checkCudaError(cudaEventElapsedTime(&duration, st, ed));");
    emit("std::cout << \" Time(ms) : \" << duration / rounds << std::endl;");

    for (auto t : tensors) {
        genTensorFree(*t);
    }
    // FIXME: Hack for bias
    for (auto &&op : graph->getOperators()) {
        if (op->getType() == Operator::Conv) {
            Tensor *t = ((ConvOp *)op)->getBias();
            if (t != nullptr)
                genTensorFree(*t);
        }
        if (op->getType() == Operator::Matmul) {
            Tensor *t = ((MatmulOp *)op)->getBias();
            if (t != nullptr)
                genTensorFree(*t);
        }
    }

    // TODO: Destroy all the descriptors

    shiftTab(-1);
    line = "}";
    emit(line);

    return render();
}

int CodeEngine::genCode(std::shared_ptr<SubGraph> &graph,
                        const std::string &filename) {
    if (check_existed(filename)) {
        std::cout << "File " << filename << " existed." << std::endl;
        // return 1;
    }

    std::string code = genCode(graph);
    std::ofstream fout(filename);
    fout << code;
    return 0;
}

int CodeEngine::clear() {
    head = "";
    main = "";
    transposeMap.clear();
    return 0;
}

int CodeEngine::shiftTab(int n) {
    if (tabs + n < 0) {
        std::cout << "invalid tab shift." << std::endl;
        return 1;
    }
    tabs += n;
    return 0;
}

int CodeEngine::emit(std::string line) {
    std::string tmp = "";
    for (int i = 0; i < tabs; i++) {
        tmp += "\t";
    }
    tmp += line + "\n";
    main += tmp;
    return 0;
}

std::string CodeEngine::render() {
    std::string code("");
    code += head;
    code += "\n";
    code += main;
    return code;
}

std::string CodeEngine::getVarName(const Tensor &t) {
    return "var_" + std::to_string(t.getHash());
}

std::string CodeEngine::getTensorDescName(const Tensor &t) {
    return "desc_tensor_" + std::to_string(t.getHash());
}

std::string CodeEngine::getFilterDescName(const Tensor &t) {
    return "desc_filter_" + std::to_string(t.getHash());
}

std::string CodeEngine::getDescName(const Operator &op) {
    return "desc_op_" + std::to_string(op.getGuid());
}

void CodeEngine::genHeader() {
    head += "#include <cudnn.h>\n";
    head += "#include <cublas_v2.h>\n";
    head += "#include <curand.h>\n";
    head += "#include <ctime>\n";
    head += "#include <cstdio>\n";
    head += "#include <iostream>\n";
    head += "#include <cub/cub.cuh>\n";
    head += "#include <custom_ops.cuh>\n";
    head += "\n";
    head +=
        "inline const char *cublasGetErrorString(cublasStatus_t error) { \\\n";
    head += "    switch (error) { \\\n";
    head += "    case CUBLAS_STATUS_SUCCESS: \\\n";
    head += "        return \" CUBLAS_STATUS_SUCCESS \"; \\\n";
    head += "    case CUBLAS_STATUS_NOT_INITIALIZED: \\\n";
    head += "        return \" CUBLAS_STATUS_NOT_INITIALIZED \"; \\\n";
    head += "    case CUBLAS_STATUS_ALLOC_FAILED: \\\n";
    head += "        return \" CUBLAS_STATUS_ALLOC_FAILED \"; \\\n";
    head += "    case CUBLAS_STATUS_INVALID_VALUE: \\\n";
    head += "        return \" CUBLAS_STATUS_INVALID_VALUE \"; \\\n";
    head += "    case CUBLAS_STATUS_ARCH_MISMATCH: \\\n";
    head += "        return \" CUBLAS_STATUS_ARCH_MISMATCH \"; \\\n";
    head += "    case CUBLAS_STATUS_MAPPING_ERROR: \\\n";
    head += "        return \" CUBLAS_STATUS_MAPPING_ERROR \"; \\\n";
    head += "    case CUBLAS_STATUS_EXECUTION_FAILED: \\\n";
    head += "        return \" CUBLAS_STATUS_EXECUTION_FAILED \"; \\\n";
    head += "    case CUBLAS_STATUS_INTERNAL_ERROR: \\\n";
    head += "        return \" CUBLAS_STATUS_INTERNAL_ERROR \"; \\\n";
    head += "    case CUBLAS_STATUS_NOT_SUPPORTED: \\\n";
    head += "        return \" CUBLAS_STATUS_NOT_SUPPORTED \"; \\\n";
    head += "    case CUBLAS_STATUS_LICENSE_ERROR: \\\n";
    head += "        return \" CUBLAS_STATUS_LICENSE_ERROR \"; \\\n";
    head += "    } \\\n";
    head += "    return \" < unknown > \"; \\\n";
    head += "}\n";
    head += "\n";
    head +=
        "inline const char *curandGetErrorString(curandStatus_t error) { \\\n";
    head += "    switch (error) { \\\n";
    head += "    case CURAND_STATUS_SUCCESS: \\\n";
    head += "        return \" CURAND_STATUS_SUCCESS \"; \\\n";
    head += "    case CURAND_STATUS_VERSION_MISMATCH: \\\n";
    head += "        return \" CURAND_STATUS_VERSION_MISMATCH \"; \\\n";
    head += "    case CURAND_STATUS_NOT_INITIALIZED: \\\n";
    head += "        return \" CURAND_STATUS_NOT_INITIALIZED \"; \\\n";
    head += "    case CURAND_STATUS_ALLOCATION_FAILED: \\\n";
    head += "        return \" CURAND_STATUS_ALLOCATION_FAILED \"; \\\n";
    head += "    case CURAND_STATUS_TYPE_ERROR: \\\n";
    head += "        return \" CURAND_STATUS_TYPE_ERROR \"; \\\n";
    head += "    case CURAND_STATUS_OUT_OF_RANGE: \\\n";
    head += "        return \" CURAND_STATUS_OUT_OF_RANGE \"; \\\n";
    head += "    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: \\\n";
    head += "        return \" CURAND_STATUS_LENGTH_NOT_MULTIPLE \"; \\\n";
    head += "    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: \\\n";
    head +=
        "        return \" CURAND_STATUS_DOUBLE_PRECISION_REQUIRED \"; \\\n";
    head += "    case CURAND_STATUS_LAUNCH_FAILURE: \\\n";
    head += "        return \" CURAND_STATUS_LAUNCH_FAILURE \"; \\\n";
    head += "    case CURAND_STATUS_PREEXISTING_FAILURE: \\\n";
    head += "        return \" CURAND_STATUS_PREEXISTING_FAILURE \"; \\\n";
    head += "    case CURAND_STATUS_INITIALIZATION_FAILED: \\\n";
    head += "        return \" CURAND_STATUS_INITIALIZATION_FAILED \"; \\\n";
    head += "    case CURAND_STATUS_ARCH_MISMATCH: \\\n";
    head += "        return \" CURAND_STATUS_ARCH_MISMATCH \"; \\\n";
    head += "    case CURAND_STATUS_INTERNAL_ERROR: \\\n";
    head += "        return \" CURAND_STATUS_INTERNAL_ERROR \"; \\\n";
    head += "    } \\\n";
    head += "    return \" < unknown > \"; \\\n";
    head += "}\n";
    head += "\n";
    head += "#define checkCudaError(call) \\\n";
    head += "{ \\\n";
    head += "    auto err = call; \\\n";
    head += "    if (cudaSuccess != err) { \\\n";
    head += "        fprintf(stderr, \"Cuda error in file '%s' in line %i : "
            "%s.\\n\", \\\n";
    head +=
        "                __FILE__, __LINE__, cudaGetErrorString(err)); \\\n";
    head += "        exit(EXIT_FAILURE); \\\n";
    head += "    } \\\n";
    head += "}\n";
    head += "\n";
    head += "#define checkCudnnError(call) \\\n";
    head += "{ \\\n";
    head += "    auto err = call; \\\n";
    head += "    if (CUDNN_STATUS_SUCCESS != err) { \\\n";
    head += "        fprintf(stderr, \"Cuda error in file '%s' in line %i : "
            "%s.\\n\", \\\n";
    head += "        __FILE__, __LINE__, cudnnGetErrorString(err)); \\\n";
    head += "        exit(EXIT_FAILURE); \\\n";
    head += "    } \\\n";
    head += "}\n";
    head += "#define checkCublasError(call) \\\n";
    head += "{ \\\n";
    head += "   auto err = call; \\\n";
    head += "   if (CUBLAS_STATUS_SUCCESS != err) { \\\n";
    head += "       fprintf(stderr, \"Cuda error in file '%s' in line %i : "
            "%s.\\n\", \\\n";
    head += "                    __FILE__, __LINE__, "
            "cublasGetErrorString(err)); \\\n";
    head += "       exit(EXIT_FAILURE); \\\n";
    head += "   } \\\n";
    head += "}\n";
    head += "#define checkCurandError(call) \\\n";
    head += "{ \\\n";
    head += "    auto err = call; \\\n";
    head += "    if (CURAND_STATUS_SUCCESS != err) { \\\n";
    head += "        fprintf(stderr, \"Cuda error in file '%s' in line %i : "
            "%s.\\n\", \\\n";
    head +=
        "                __FILE__, __LINE__, curandGetErrorString(err)); \\\n";
    head += "        exit(EXIT_FAILURE); \\\n";
    head += "    } \\\n";
    head += "}\n";

    head += "\n/* online_softmax: cub is required */\n";
    head += "struct __align__(8) MD\n";
    head += "{\n";
    head += "    float m;\n";
    head += "    float d;\n";
    head += "};\n";
    head += "\n";
    head += "__device__ __forceinline__ MD reduce_md_op(MD a, MD b)\n";
    head += "{\n";
    head += "    bool a_bigger = (a.m > b.m);\n";
    head += "    MD bigger_m = a_bigger ? a : b;\n";
    head += "    MD smaller_m = a_bigger ? b : a;\n";
    head += "    MD res;\n";
    head += "    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - "
            "bigger_m.m);\n";
    head += "    res.m = bigger_m.m;\n";
    head += "    return res;\n";
    head += "}\n";
    head += "\n";
    head += "template<int THREADBLOCK_SIZE>\n";
    head += "__launch_bounds__(THREADBLOCK_SIZE)\n";
    head += "__global__ void online_softmax(\n";
    head += "    const float * __restrict x,\n";
    head += "    float * __restrict y,\n";
    head += "    int V)\n";
    head += "{\n";
    head += "    int thread_id = threadIdx.x;\n";
    head += "    int vector_id = blockIdx.x;\n";
    head += "\n";
    head += "    // reposition x and y to data for the current vector\n";
    head += "    x += vector_id * V;\n";
    head += "    y += vector_id * V;\n";
    head += "\n";
    head += "    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;\n";
    head += "\n";
    head += "    __shared__ typename BlockReduce::TempStorage temp_storage;\n";
    head += "    __shared__ MD md_total;\n";
    head += "\n";
    head += "    MD md_partial;\n";
    head += "    md_partial.m = -FLT_MAX;\n";
    head += "    md_partial.d = 0.0F;\n";
    head += "    for(int elem_id = thread_id; elem_id < V; elem_id += "
            "THREADBLOCK_SIZE)\n";
    head += "    {\n";
    head += "        MD new_elem;\n";
    head += "        new_elem.m = x[elem_id];\n";
    head += "        new_elem.d = 1.0F;\n";
    head += "        md_partial = reduce_md_op(md_partial, new_elem);\n";
    head += "    }\n";
    head += "\n";
    head += "    MD md = BlockReduce(temp_storage).Reduce(md_partial, "
            "reduce_md_op);\n";
    head += "    if (thread_id == 0)\n";
    head += "        md_total = md;\n";
    head += "    __syncthreads();\n";
    head += "\n";
    head += "    float d_total_inverse = __fdividef(1.0F, md_total.d);\n";
    head += "    for(int elem_id = thread_id; elem_id < V; elem_id += "
            "THREADBLOCK_SIZE)\n";
    head += "        y[elem_id] = __expf(x[elem_id] - md_total.m) * "
            "d_total_inverse;\n";
    head += "}\n";
    head += "/* online_softmax: cub is required */\n\n";
    head += "#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)\n";
    head += "void __cudaCheckError(const char *file, const int line) {\n";
    head += "    cudaError err = cudaGetLastError();\n";
    head += "    if (cudaSuccess != err) {\n";
    head += "        std::cout << \"[ERROR] \" << file << \"::\" << line\n";
    head += "                  << \": cudaCheckError() failed. \" << "
            "cudaGetErrorString(err)\n";
    head += "                  << std::endl;\n";
    head += "        exit(-1);\n";
    head += "    }\n";
    head += "    return;\n";
    head += "}\n\n";
    head += "__global__ void kernel_transpose_3d(float *dst, float *src, const "
            "int b, const int m, const int n) {\n";
    head += "    float buf[32];\n";
    head += "    int warp_id = threadIdx.x / 32;\n";
    head += "    int lane_id = threadIdx.x % 32;\n";
    head += "    int nm = m / 32;\n";
    head += "    int nn = n / 32;\n";
    head += "    int nt = b * nm * nn;\n";
    head += "    int base = blockIdx.x * 4 + warp_id;\n";
    head += "    int step = gridDim.x * 4;\n";
    head += "\n";
    head += "    for (int idx = base; idx < nt; idx += step) {\n";
    head += "        int ib = idx;\n";
    head += "        int in = ib % nn;\n";
    head += "        ib /= nn;\n";
    head += "        int im = ib % nm;\n";
    head += "        ib /= nm;\n";
    head += "        int offset_src = ib * m * n + im * 32 * n + in * 32;\n";
    head += "        int offset_dst = ib * m * n + in * 32 * m + im * 32;\n";
    head += "#pragma unroll\n";
    head += "        for (int i = 0; i < 32; i++) {\n";
    head += "            int j = (i + lane_id) % 32;\n";
    head += "            buf[i] = src[offset_src + i * n + j];\n";
    head += "        }\n";
    head += "#pragma unroll\n";
    head += "        for (int j = 0; j < 32; j++) {\n";
    head += "            int i = (j + 32 - lane_id) % 32;\n";
    head += "            dst[offset_dst + j * m + i] = buf[i];\n";
    head += "        }\n";
    head += "    }\n";
    head += "    return;\n";
    head += "}\n";
}

Dim CodeEngine::getDim(const Tensor &t) {
    Dim dim = t.getDims();
    assert(perfEngine != nullptr);
    if (perfEngine->withPenalty()) {
        Dim &&penalty = t.getPenalty();
        assert(penalty.size() == dim.size());
        for (size_t i = 0, iEnd = dim.size(); i < iEnd; i++) {
            assert(penalty[i] >= 0);
            dim[i] += penalty[i];
        }
    }
    return dim;
}

size_t CodeEngine::getTensorNElem(const Tensor &t) {
    size_t size = 1;
    for (auto dim : getDim(t)) {
        size *= dim;
    }
    return size;
}

size_t CodeEngine::getTensorSize(const Tensor &t) {
    assert(t.getDType() == Tensor::Float32 || t.getDType() == Tensor::Int32);
    return 4 * getTensorNElem(t);
}

void CodeEngine::genTensorAlloc(const Tensor &t, bool isConvBias) {
    std::string line;
    std::string var_name = getVarName(t);
    Dim dims = getDim(t);
    int size = getTensorSize(t);

    if (isConvBias) {
        assert(dims.size() == 1);
        dims = {1, dims[0], 1, 1};
    }

    line = fmt::format("// {} ( Dim: ", var_name);
    for (auto dim : dims) {
        line += fmt::format("{} ", dim);
    }
    line += ")";
    emit(line);
    switch (t.getDType()) {
    case Tensor::Float32:
        emit(fmt::format("float *{} = 0;", var_name));
        emit(fmt::format("checkCudaError(cudaMalloc((void **) &{}, {}));",
                         var_name, std::to_string(size)));
        emit(
            fmt::format("checkCurandError(curandGenerateUniform(gen, {}, {}));",
                        var_name, std::to_string(getTensorNElem(t))));
        break;
    case Tensor::Int32:
        emit(fmt::format("int32_t *{} = 0;", var_name));
        emit(fmt::format("checkCudaError(cudaMalloc((void **) &{}, {}));",
                         var_name, std::to_string(size)));
        // Integers are used as indices, so no random
        emit(fmt::format("checkCudaError(cudaMemset({}, 0, {}));", var_name,
                         std::to_string(size)));
        break;
    default:
        assert(false);
    }

    std::vector<int> paddedDim = dims;
    while (paddedDim.size() < 4) {
        paddedDim.insert(paddedDim.begin(), 1);
    }
    assert(paddedDim.size() == 4);

    emit(fmt::format("cudnnTensorDescriptor_t {};", getTensorDescName(t)));
    emit(fmt::format("checkCudnnError(cudnnCreateTensorDescriptor(&{}));",
                     getTensorDescName(t)));
    if (t.getType() == Tensor::Weight) {
        emit(fmt::format("cudnnFilterDescriptor_t {};", getFilterDescName(t)));
        emit(fmt::format("checkCudnnError(cudnnCreateFilterDescriptor(&{}));",
                         getFilterDescName(t)));
    }

    std::string dtype;
    switch (t.getDType()) {
    case Tensor::Float32:
        dtype = "CUDNN_DATA_FLOAT";
        break;
    case Tensor::Int32:
        dtype = "CUDNN_DATA_INT32";
        break;
    default:
        assert(false);
    }

    line = fmt::format("checkCudnnError(cudnnSetTensor4dDescriptor({}, {}, "
                       "CUDNN_TENSOR_NCHW, {}, {}, {}, {}));",
                       getTensorDescName(t), dtype, paddedDim[0], paddedDim[1],
                       paddedDim[2], paddedDim[3]);
    emit(line);

    if (t.getType() == Tensor::Weight) {
        line = fmt::format("checkCudnnError(cudnnSetFilter4dDescriptor({}, {}, "
                           "CUDNN_TENSOR_NCHW, {}, {}, {}, {}));",
                           getFilterDescName(t), dtype, paddedDim[0],
                           paddedDim[1], paddedDim[2], paddedDim[3]);
        emit(line);
    }
}

void CodeEngine::genTensorFree(const Tensor &t) {
    emit(fmt::format("checkCudnnError(cudnnDestroyTensorDescriptor({}));",
                     getTensorDescName(t)));
    if (t.getType() == Tensor::Weight) {
        emit(fmt::format("checkCudnnError(cudnnDestroyFilterDescriptor({}));",
                         getFilterDescName(t)));
    }
    emit(fmt::format("cudaFree({});", getVarName(t)));
}

void CodeEngine::genDesc(const Operator &op) {
    emit("");
    switch (op.getType()) {
    case Operator::Conv:
        genConvDesc(static_cast<const ConvOp &>(op));
        break;
    case Operator::Matmul:
        genMatmulDesc(static_cast<const MatmulOp &>(op));
        break;
    case Operator::Pad:
        genPadDesc(static_cast<const PadOp &>(op));
        break;
    case Operator::Slice:
        genSliceDesc(static_cast<const SliceOp &>(op));
        break;
    case Operator::Activation:
        genActivationDesc(static_cast<const ActivationOp &>(op));
        break;
    case Operator::AvgPool:
        genAvgPoolDesc(static_cast<const AvgPoolOp &>(op));
        break;
    case Operator::MaxPool:
        genMaxPoolDesc(static_cast<const MaxPoolOp &>(op));
        break;
    case Operator::Add:
        genAddDesc(static_cast<const AddOp &>(op));
        break;
    case Operator::Mul:
        genMulDesc(static_cast<const MulOp &>(op));
        break;
    case Operator::Transpose:
        genTransposeDesc(static_cast<const TransposeOp &>(op));
        break;
    case Operator::Gather:
        genGatherDesc(static_cast<const GatherOp &>(op));
        break;
    case Operator::Split:
        genSplitDesc(static_cast<const SplitOp &>(op));
        break;
    case Operator::Concat:
        genConcatDesc(static_cast<const ConcatOp &>(op));
        break;
    case Operator::Extend:
        genExtendDesc(static_cast<const ExtendOp &>(op));
        break;
    case Operator::Reshape:
        genReshapeDesc(static_cast<const ReshapeOp &>(op));
        break;
    case Operator::Softmax:
        genSoftmaxDesc(static_cast<const SoftmaxOp &>(op));
        break;
    case Operator::MemBound:
        genMemBoundDesc(static_cast<const MemBoundOp &>(op));
        break;
    case Operator::ConvTrans:
        genConvTransDesc(static_cast<const ConvTransOp &>(op));
        break;
    case Operator::G2BMM:
        genG2BMMDesc(static_cast<const G2BMMOp &>(op));
        break;
    case Operator::GBMML:
        genGBMMLDesc(static_cast<const GBMMLOp &>(op));
        break;
    case Operator::BatchNorm:
        genBatchNormDesc(static_cast<const BatchNormOp &>(op));
        break;
    default:
        op.print();
        assert(false);
    }
}

void CodeEngine::genCompute(const Operator &op) {
    emit("");
    switch (op.getType()) {
    case Operator::Conv:
        genConvCompute(static_cast<const ConvOp &>(op));
        break;
    case Operator::Matmul:
        genMatmulCompute(static_cast<const MatmulOp &>(op));
        break;
    case Operator::Pad:
        genPadCompute(static_cast<const PadOp &>(op));
        break;
    case Operator::Slice:
        genSliceCompute(static_cast<const SliceOp &>(op));
        break;
    case Operator::Activation:
        genActivationCompute(static_cast<const ActivationOp &>(op));
        break;
    case Operator::AvgPool:
    case Operator::MaxPool:
        genPoolCompute(op);
        break;
    case Operator::Add:
        genAddCompute(static_cast<const AddOp &>(op));
        break;
    case Operator::Mul:
        genMulCompute(static_cast<const MulOp &>(op));
        break;
    case Operator::Transpose:
        genTransposeCompute(static_cast<const TransposeOp &>(op));
        break;
    case Operator::Gather:
        genGatherCompute(static_cast<const GatherOp &>(op));
        break;
    case Operator::Split:
        genSplitCompute(static_cast<const SplitOp &>(op));
        break;
    case Operator::Concat:
        genConcatCompute(static_cast<const ConcatOp &>(op));
        break;
    case Operator::Extend:
        genExtendCompute(static_cast<const ExtendOp &>(op));
        break;
    case Operator::Reshape:
        genReshapeCompute(static_cast<const ReshapeOp &>(op));
        break;
    case Operator::Softmax:
        genSoftmaxCompute(static_cast<const SoftmaxOp &>(op));
        break;
    case Operator::MemBound:
        genMemBoundCompute(static_cast<const MemBoundOp &>(op));
        break;
    case Operator::ConvTrans:
        genConvTransCompute(static_cast<const ConvTransOp &>(op));
        break;
    case Operator::G2BMM:
        genG2BMMCompute(static_cast<const G2BMMOp &>(op));
        break;
    case Operator::GBMML:
        genGBMMLCompute(static_cast<const GBMMLOp &>(op));
        break;
    case Operator::BatchNorm:
        genBatchNormCompute(static_cast<const BatchNormOp &>(op));
        break;
    default:
        op.print();
        assert(false);
    }
}

void CodeEngine::genConvDesc(const ConvOp &op) {
    emit(fmt::format("cudnnConvolutionDescriptor_t {};", getDescName(op)));
    emit(fmt::format("checkCudnnError(cudnnCreateConvolutionDescriptor(&{}));",
                     getDescName(op)));

    auto line = fmt::format(
        "checkCudnnError(cudnnSetConvolution2dDescriptor({}, {}, {}, {}, {}, "
        "{}, {}, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));",
        getDescName(op), op.getPh(), op.getPw(), op.getSh(), op.getSw(),
        op.getDh(), op.getDw());
    emit(line);

    int arg_g = getDim(*op.getInputs()[0])[1] / getDim(*op.getInputs()[1])[1];
    if (arg_g > 1) {
        emit(fmt::format(
            "checkCudnnError(cudnnSetConvolutionGroupCount({}, {}));",
            getDescName(op), arg_g));
    }

    if (op.getAct() != Operator::None) {
        emit(fmt::format("cudnnActivationDescriptor_t {}_act ;",
                         getDescName(op)));
        emit(fmt::format(
            "checkCudnnError(cudnnCreateActivationDescriptor(&{}_act));",
            getDescName(op)));
        auto act_line =
            fmt::format("checkCudnnError(cudnnSetActivationDescriptor({}_act, "
                        "{}, CUDNN_NOT_PROPAGATE_NAN, 0));",
                        getDescName(op), actToStr(op.getAct()));
        emit(act_line);
    }
}

void CodeEngine::genConvTransDesc(const ConvTransOp &op) {
    emit(fmt::format("cudnnConvolutionDescriptor_t {};", getDescName(op)));
    emit(fmt::format("checkCudnnError(cudnnCreateConvolutionDescriptor(&{}));",
                     getDescName(op)));
    std::string line = fmt::format(
        "checkCudnnError(cudnnSetConvolution2dDescriptor({}, {}, {}, {}, {}, "
        "{}, {}, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));",
        getDescName(op), op.getPh(), op.getPw(), op.getSh(), op.getSw(),
        op.getDh(), op.getDw());
    emit(line);
    int arg_g = getDim(*op.getInputs()[0])[1] / getDim(*op.getInputs()[1])[1];
    if (arg_g > 1) {
        emit(fmt::format(
            "checkCudnnError(cudnnSetConvolutionGroupCount({}, {}));",
            getDescName(op), arg_g));
    }

    if (op.getAct() != Operator::None) {
        emit(fmt::format("cudnnActivationDescriptor_t {}_act;",
                         getDescName(op)));
        emit(fmt::format(
            "checkCudnnError(cudnnCreateActivationDescriptor(&{}_act));",
            getDescName(op)));
        std::string line =
            fmt::format("checkCudnnError(cudnnSetActivationDescriptor({}_act, "
                        "{}, CUDNN_NOT_PROPAGATE_NAN, 0));",
                        getDescName(op), actToStr(op.getAct()));
        // NOT_PROPAGATE_NAN is requierd by
        // cudnnConvolotionBiasActivationForward
        emit(line);
    }
}

void CodeEngine::genConvCompute(const ConvOp &op) {
    std::string alpha = fmt::format("alpha_{}", std::to_string(op.getGuid()));
    std::string beta = fmt::format("beta_{}", std::to_string(op.getGuid()));
    emit(fmt::format("float {} = 1.0f, {} = 0.0f;", alpha, beta));

    assert(perfEngine->getOpPerf(Operator::Conv,
                                 op.getArgs(perfEngine->withPenalty())) <
           INFINITY);
    int algo =
        (int)perfEngine->getConvAlgo(op.getArgs(perfEngine->withPenalty()));
    if (op.getAct() == Operator::None && op.getBias() == nullptr) {
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionForward(cudnn, &{}, {}, {}, {}, "
            "{}, {}, {}, (cudnnConvolutionFwdAlgo_t){}, wsData, wsSize, &{}, "
            "{}, {}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);

    } else if (op.getAct() == Operator::None && op.getBias() != nullptr) {
        // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is
        // enabled with CUDNN_ACTIVATION_IDENTITY.
        // So, fallback to 2 seperated calls
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionForward(cudnn, &{}, {}, {}, {}, "
            "{}, {}, {}, (cudnnConvolutionFwdAlgo_t){}, wsData, wsSize, &{}, "
            "{}, {}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
        line = fmt::format(
            "checkCudnnError(cudnnAddTensor(cudnn, &{}, {}, {}, &{}, {}, {}));",
            alpha, getTensorDescName(*op.getBias()), getVarName(*op.getBias()),
            beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
    } else if (op.getAct() != Operator::None && op.getBias() == nullptr) {
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionForward(cudnn, &{}, {}, {}, {}, "
            "{}, {}, (cudnnConvolutionFwdAlgo_t){}, wsData, wsSize, &{}, {}, "
            "{}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
        emit(fmt::format("{} = 1.0f;", beta));
        line = fmt::format("checkCudnnError(cudnnActivationForward(cudnn, "
                           "{}_act, &{}, {}, &{}, {}, {}));",
                           getDescName(op), alpha,
                           getTensorDescName(*op.getOutputs()[0]),
                           getVarName(*op.getOutputs()[0]), beta,
                           getTensorDescName(*op.getOutputs()[0]),
                           getVarName(*op.getOutputs()[0]));
        emit(line);
    } else if (op.getAct() != Operator::None && op.getBias() != nullptr) {
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionBiasActivationForward(cudnn, &{}, "
            "{}, {}, {}, {}, {}, (cudnnConvolutionFwdAlgo_t){}, wsData, "
            "wsSize, &{}, {}, {}, {}, {}, {}, {}_act, {}, {}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]), getTensorDescName(*op.getBias()),
            getVarName(*op.getBias()), getDescName(op),
            getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
    } else {
        assert(false);
    }
}

void CodeEngine::genConvTransCompute(const ConvTransOp &op) {
    std::string alpha = fmt::format("alpha_{}", std::to_string(op.getGuid()));
    std::string beta = fmt::format("beta_{}", std::to_string(op.getGuid()));
    emit(fmt::format("float {} = 1.0f, {} = 0.0f;", alpha, beta));

    assert(perfEngine->getOpPerf(Operator::ConvTrans,
                                 op.getArgs(perfEngine->withPenalty())) <
           INFINITY);
    int algo = (int)perfEngine->getConvTransAlgo(
        op.getArgs(perfEngine->withPenalty()));
    if (op.getAct() == Operator::None && op.getBias() == nullptr) {
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionBackwardData(cudnn, &{}, {}, {}, "
            "{}, {}, {}, (cudnnConvolutionBwdDataAlgo_t){}, wsData, wsSize, "
            "&{}, {}, {}));",
            alpha, getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]),
            getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
    } else if (op.getAct() == Operator::None && op.getBias() != nullptr) {
        // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is
        // enabled with CUDNN_ACTIVATION_IDENTITY.
        // So, fallback to 2 seperated calls
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionBackwardData(cudnn, &{}, {}, {}, "
            "{}, {}, {}, (cudnnConvolutionBwdDataAlgo_t){}, wsData, wsSize, "
            "&{}, {}, {}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
        emit(fmt::format("{} = 1.0f;", beta));
        line = fmt::format(
            "checkCudnnError(cudnnAddTensor(cudnn, &{}, {}, {}, &{}, {}, {}));",
            alpha, getTensorDescName(*op.getBias()), getVarName(*op.getBias()),
            beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);

    } else if (op.getAct() != Operator::None && op.getBias() == nullptr) {
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionBackwardData(cudnn, &{}, {}, {}, "
            "{}, {}, {}, (cudnnConvolutionBwdDataAlgo_t){}, wsData, wsSize, "
            "&{}, {}, {}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
        emit(fmt::format("{} = 1.0f;", beta));
        line = fmt::format("checkCudnnError(cudnnActivationBackward(cudnn, "
                           "{}_act, &{}, {}, {}, &{}, {}, {}));",
                           getDescName(op), alpha,
                           getTensorDescName(*op.getOutputs()[0]),
                           getVarName(*op.getOutputs()[0]), beta,
                           getTensorDescName(*op.getOutputs()[0]),
                           getVarName(*op.getOutputs()[0]));
        emit(line);
    } else if (op.getAct() != Operator::None && op.getBias() != nullptr) {
        assert(((void)("No corresponding cuDNN kernels?"), false));
        std::string line = fmt::format(
            "checkCudnnError(cudnnConvolutionBiasActivationForward(cudnn, &{}, "
            "{}, {}, {}, {}, {}, (cudnnConvolutionFwdAlgo_t){}, wsData, "
            "wsSize, &{}, {}, {}, {}, {}, {}_act, {}, {}));",
            alpha, getTensorDescName(*op.getInputs()[0]),
            getVarName(*op.getInputs()[0]),
            getFilterDescName(*op.getInputs()[1]),
            getVarName(*op.getInputs()[1]), getDescName(op),
            std::to_string(algo), beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]), getTensorDescName(*op.getBias()),
            getVarName(*op.getBias()), getDescName(op),
            getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
    } else {
        assert(false);
    }
}

void CodeEngine::genMatmulDesc(const MatmulOp &op) {
    if (op.getAct() != Operator::None) {
        emit(fmt::format("cudnnActivationDescriptor_t {}_act;",
                         getDescName(op)));
        emit(fmt::format(
            "checkCudnnError(cudnnCreateActivationDescriptor(&{}_act));",
            getDescName(op)));
        std::string line =
            fmt::format("checkCudnnError(cudnnSetActivationDescriptor({}_act, "
                        "{}, CUDNN_NOT_PROPAGATE_NAN, 0));",
                        getDescName(op), actToStr(op.getAct()));
        emit(line);
    }
}

void CodeEngine::genMatmulCompute(const MatmulOp &op) {
    auto A = op.getInputs()[0], B = op.getInputs()[1];
    auto &&dimA = getDim(*A), &&dimB = getDim(*B);
    auto b = dimA[0];
    auto m = op.getTransA() ? dimA[2] : dimA[1];
    auto n = op.getTransB() ? dimB[1] : dimB[2];
    auto k = op.getTransA() ? dimA[1] : dimA[2];
    const int lda = op.getTransA() ? m : k, ldb = op.getTransB() ? k : n,
              ldc = n;

    std::string alpha = fmt::format("alpha_{}", std::to_string(op.getGuid()));
    std::string beta = fmt::format("beta_{}", std::to_string(op.getGuid()));
    emit(fmt::format("float {} = 1.0f, {} = 0.0f;", alpha, beta));

    std::string opB_str = op.getTransB() ? "CUBLAS_OP_T" : "CUBLAS_OP_N"; // opB
    std::string opA_str = op.getTransA() ? "CUBLAS_OP_T" : "CUBLAS_OP_N"; // opA
    std::string b_str = getVarName(*op.getInputs()[1]);                   // B
    std::string a_str = getVarName(*op.getInputs()[0]);                   // A
    std::string c_str = getVarName(*op.getOutputs()[0]);                  // C

    std::string line = fmt::format(
        "cublasGemmStridedBatchedEx({}, {}, {}, {}, {}, {}, &{}, {}, {}, {}, "
        "{}, {}, {}, {}, {}, &{}, {}, {}, {}, {}, {}, {}, "
        "(cublasGemmAlgo_t){});",
        "cublas", opB_str, opA_str, std::to_string(n), std::to_string(m),
        std::to_string(k), alpha, b_str, "CUDA_R_32F", std::to_string(ldb),
        std::to_string(k * n), a_str, "CUDA_R_32F", std::to_string(lda),
        std::to_string(m * k), beta, c_str, "CUDA_R_32F", std::to_string(ldc),
        std::to_string(m * n), std::to_string(b), "CUDA_R_32F",
        std::to_string((int)perfEngine->getMatmulAlgo(op.getArgs())));
    emit(line);

    if (op.getAct() != Operator::None) {
        emit(fmt::format("{} = 1.0f;", beta));
        line = fmt::format("checkCudnnError(cudnnActivationForward(cudnn, "
                           "{}_act, &{}, {}, {}, &{}, {}, {}));",
                           getDescName(op), alpha,
                           getTensorDescName(*op.getOutputs()[0]),
                           getVarName(*op.getOutputs()[0]), beta,
                           getTensorDescName(*op.getOutputs()[0]),
                           getVarName(*op.getOutputs()[0]));
        emit(line);
    }

    if (op.getBias() != nullptr) {
        emit(fmt::format("{} = 1.0f;", beta));
        line = fmt::format(
            "checkCudnnError(cudnnAddTensor(cudnn, &{}, {}, {}, &{}, {}, {}));",
            alpha, getTensorDescName(*op.getBias()), getVarName(*op.getBias()),
            beta, getTensorDescName(*op.getOutputs()[0]),
            getVarName(*op.getOutputs()[0]));
        emit(line);
    }
}

void CodeEngine::genPadDesc(const PadOp &op) {
    emit(fmt::format("cudnnTensorTransformDescriptor_t {};", getDescName(op)));
    emit(fmt::format(
        "checkCudnnError(cudnnCreateTensorTransformDescriptor(&{}));",
        getDescName(op)));

    emit("{"); // so the arrays will be locals
    shiftTab(1);

    std::string line = fmt::format(
        "int padBefore[] = {{}, {}, {}, {}}", std::to_string(op.getBegin()[0]),
        std::to_string(op.getBegin()[1]), std::to_string(op.getBegin()[2]),
        std::to_string(op.getBegin()[3]));
    emit(line);
    line = fmt::format(
        "int padAfter[] = {{}, {}, {}, {}}", std::to_string(op.getEnd()[0]),
        std::to_string(op.getEnd()[1]), std::to_string(op.getEnd()[2]),
        std::to_string(op.getEnd()[3]));
    emit(line);
    line = fmt::format("checkCudnnError(cudnnSetTensorTransformDescriptor({}, "
                       "4, CUDNN_TENSOR_NCHW, padBefore, padAfter, nullptr, "
                       "CUDNN_TRANSFORM_FOLD));",
                       getDescName(op));
    emit(line);

    shiftTab(-1);
    emit("}");
}

void CodeEngine::genPadCompute(const PadOp &op) {
    std::string alpha = fmt::format("alpha_{}", std::to_string(op.getGuid()));
    std::string beta = fmt::format("beta_{}", std::to_string(op.getGuid()));
    emit(fmt::format("float {} = 1.0f, {} = 0.0f;", alpha, beta));
    std::string line = fmt::format("checkCudnnError(cudnnTransformTensorEx("
                                   "cudnn, {}, &{}, {}, {}, &{}, {}, {}));",
                                   getDescName(op), alpha,
                                   getTensorDescName(*op.getInputs()[0]),
                                   getVarName(*op.getInputs()[0]), beta,
                                   getTensorDescName(*op.getOutputs()[0]),
                                   getVarName(*op.getOutputs()[0]));
    emit(line);
}

void CodeEngine::genSliceDesc(const SliceOp &op) {
    // Empty
}

void CodeEngine::genSliceCompute(const SliceOp &op) {
    const Dim &inDim = getDim(*op.getInputs()[0]);
    const Dim &outDim = getDim(*op.getOutput());
    size_t nDim = outDim.size();
    std::string lambda = "lambda ";
    for (size_t i = 0; i < nDim; i++) {
        lambda += std::string(1, 'a' + i) + (i < nDim - 1 ? ", " : "");
    }
    lambda += ": I[0][";
    for (size_t i = 0; i < nDim; i++) {
        lambda += std::string(1, 'a' + i) + " - " +
                  std::to_string(op.getBegin()[i]) + (i < nDim - 1 ? ", " : "");
    }
    lambda += "]";

    std::string func = "slice_" + std::to_string(op.getGuid());
    std::string input = getVarName(*op.getInputs()[0]);
    std::string output = getVarName(*op.getOutputs()[0]);

    auto res =
        getTVMCode({inDim}, {"float32"}, outDim, lambda, func, {input}, output);

    head += "\n" + res.first + "\n";
    main += "\n" + res.second + "\n";
}

void CodeEngine::genActivationDesc(const ActivationOp &op) {
    emit(fmt::format("cudnnActivationDescriptor_t {};", getDescName(op)));
    emit(fmt::format("checkCudnnError(cudnnCreateActivationDescriptor(&{}));",
                     getDescName(op)));
    std::string line =
        fmt::format("checkCudnnError(cudnnSetActivationDescriptor({}, {}, "
                    "CUDNN_NOT_PROPAGATE_NAN, 0));",
                    getDescName(op), actToStr(op.getActType()));
    emit(line);
}

void CodeEngine::genActivationCompute(const ActivationOp &op) {
    std::string alpha = fmt::format("alpha_{}", std::to_string(op.getGuid()));
    std::string beta = fmt::format("beta_{}", std::to_string(op.getGuid()));
    emit(fmt::format("float {} = 1.0f, {} = 0.0f;", alpha, beta));
    std::string line = fmt::format("checkCudnnError(cudnnActivationForward("
                                   "cudnn, {}, &{}, {}, {}, &{}, {}, &{}));",
                                   getDescName(op), alpha,
                                   getTensorDescName(*op.getInputs()[0]),
                                   getVarName(*op.getInputs()[0]), beta,
                                   getTensorDescName(*op.getOutputs()[0]),
                                   getVarName(*op.getOutputs()[0]));
    emit(line);
}

void CodeEngine::genMaxPoolDesc(const MaxPoolOp &op) {
    emit(fmt::format("cudnnPoolingDescriptor_t {};", getDescName(op)));
    emit(fmt::format("checkCudnnError(cudnnCreatePoolingDescriptor(&{}));",
                     getDescName(op)));
    std::string line = fmt::format(
        "checkCudnnError(cudnnSetPooling2dDescriptor({}, CUDNN_POOLING_MAX, "
        "CUDNN_NOT_PROPAGATE_NAN, {}, {}, {}, {}, {}, {}));",
        getDescName(op), op.getKh(), op.getKw(), op.getPh(), op.getPw(),
        op.getSh(), op.getSw());
    assert(op.getDh() == 1);
    assert(op.getDw() == 1);
    emit(line);
}

void CodeEngine::genAvgPoolDesc(const AvgPoolOp &op) {
    emit(fmt::format("cudnnPoolingDescriptor_t {};", getDescName(op)));
    emit(fmt::format("checkCudnnError(cudnnCreatePoolingDescriptor(&{}));",
                     getDescName(op)));
    std::string line =
        fmt::format("checkCudnnError(cudnnSetPooling2dDescriptor("
                    "&{}, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, "
                    "CUDNN_NOT_PROPAGATE_NAN, "
                    "{}, {}, {}, {}, {}, {}));",
                    getDescName(op), op.getKh(), op.getKw(), op.getPh(),
                    op.getPw(), op.getSh(), op.getSw());
    emit(line);
}

void CodeEngine::genPoolCompute(const Operator &op) {
    std::string alpha = fmt::format("alpha_{}", op.getGuid());
    std::string beta = fmt::format("beta_{}", op.getGuid());
    emit(fmt::format("float {} = 1.0f, {} = 0.0f;", alpha, beta));
    std::string line = fmt::format(
        "checkCudnnError(cudnnPoolingForward(cudnn, {}, "
        "&{}, {}, {}, &{}, {}, {}));",
        getDescName(op), alpha, getTensorDescName(*op.getInputs()[0]),
        getVarName(*op.getInputs()[0]), beta,
        getTensorDescName(*op.getOutputs()[0]),
        getVarName(*op.getOutputs()[0]));
    emit(line);
}

void CodeEngine::genAddDesc(const AddOp &op) {
    // Empty
}

void CodeEngine::genAddCompute(const AddOp &op) {
    // TODO inplace operation assignment is not correct
    std::string alpha = fmt::format("alpha_{}", op.getGuid());
    emit(fmt::format("float {} = 1.0f;", alpha));
    std::string line = fmt::format(
        "checkCublasError(cublasSaxpy(cublas, {}, &{}, {}, {}, 1, {}, 1));",
        op.getInputs()[1]->size(), alpha, getVarName(*op.getInputs()[0]),
        getVarName(*op.getInputs()[1]));
    emit(line);
}

#if 0
void CodeEngine::genAddCompute(const AddOp &op) {
    Dim dimO = {(int)getTensorNElem(*op.getOutput())};
    std::vector<Dim> inDims;
    std::vector<std::string> inDTypes, inNames;
    std::string lambda = "lambda a: ";
    for (size_t inId = 0, inNum = op.getInputs().size(); inId < inNum; inId++) {
        auto &&in = op.getInputs()[inId];
        int inSize = getTensorNElem(*in);
        lambda += "I[" + std::to_string(inId) + "][a % " +
                  std::to_string(inSize) + "]" +
                  (inId < inNum - 1 ? " + " : "");
        inDims.emplace_back(Dim{inSize});
        inDTypes.emplace_back("float32");
        inNames.emplace_back(getVarName(*in));
    }
    std::string func = "add_" + std::to_string(op.getGuid());
    std::string output = getVarName(*op.getOutput());

    auto res =
        getTVMCode(inDims, inDTypes, dimO, lambda, func, {inNames}, output);

    head += "\n" + res.first + "\n";
    main += "\n" + res.second + "\n";
}
#endif

void CodeEngine::genMulDesc(const MulOp &op) {
    emit(fmt::format("cudnnOpTensorDescriptor_t {};", getDescName(op)));
    emit(fmt::format("checkCudnnError(cudnnCreateOpTensorDescriptor(&{}));",
                     getDescName(op)));
    std::string line = fmt::format(
        "checkCudnnError(cudnnSetOpTensorDescriptor({}, CUDNN_OP_TENSOR_MUL, "
        "CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));",
        getDescName(op));
    emit(line);
}

void CodeEngine::genMulCompute(const MulOp &op) {
    std::string alpha1 = fmt::format("alpha1_{}", op.getGuid());
    std::string alpha2 = fmt::format("alpha2_{}", op.getGuid());
    std::string beta = fmt::format("beta_{}", op.getGuid());
    emit(fmt::format("float {} = 1.0f, {} = 1.0f, {} = 0.0f;", alpha1, alpha2,
                     beta));
    std::string line = fmt::format(
        "checkCudnnError(cudnnOpTensor(cudnn, {}, &{}, {}, {}, &{}, {}, "
        "{}, &{}, {}, {}));",
        getDescName(op), alpha1, getTensorDescName(*op.getInputs()[0]),
        getVarName(*op.getInputs()[0]), alpha2,
        getTensorDescName(*op.getInputs()[1]), getVarName(*op.getInputs()[1]),
        beta, getTensorDescName(*op.getOutputs()[0]),
        getVarName(*op.getOutputs()[0]));
    emit(line);
    for (size_t i = 2, iEnd = op.getInputs().size(); i < iEnd; i++) {
        std::string line = fmt::format(
            "checkCudnnError(cudnnOpTensor("
            "cudnn, {}, &{}, {}, {}, &{}, {}, {}, &{}, {}, {}));",
            getDescName(op), alpha1, getTensorDescName(*(op.getInputs()[i])),
            getVarName(*(op.getInputs()[i])), alpha2,
            getTensorDescName(*(op.getOutputs()[0])),
            getVarName(*(op.getOutputs()[0])), beta,
            getTensorDescName(*(op.getOutputs()[0])),
            getVarName(*(op.getOutputs()[0])));
        emit(line);
    }
}

void CodeEngine::genTransposeDesc(const TransposeOp &op) {
    // Empty
}

void CodeEngine::genTransposeCompute(const TransposeOp &op) {
    if (!transposeMap.count(&op)) {
        transposeMap[&op].reset(new std::vector<const TransposeOp *>());
    }
    transposeMap.at(&op)->emplace_back(&op);
    if (op.getSuccessors().size() == 1 &&
        op.getSuccessors()[0]->getPredecessors().size() == 1 &&
        op.getSuccessors()[0]->isTransposeOp()) {
        transposeMap[static_cast<TransposeOp *>(op.getSuccessors()[0])] =
            transposeMap.at(&op);
        transposeMap.erase(&op);
        return;
    }

    const std::vector<const TransposeOp *> &ops = *transposeMap.at(&op);

    std::string func = "transpose_" + std::to_string(op.getGuid());
    std::string input = getVarName(*ops.front()->getInputs()[0]);
    std::string output = getVarName(*ops.back()->getOutputs()[0]);

    auto res = genTranspose(ops, func, input, output);

    std::string comment = "// ";
    for (auto &&op : ops) {
        comment += op->toString() + " -> ";
    }

    head += "\n" + res.first + "\n";
    main += comment + "\n" + res.second + "\n";

    transposeMap.erase(&op);
}

void CodeEngine::genGatherDesc(const GatherOp &op) {
    // Empty
}

void CodeEngine::genGatherCompute(const GatherOp &op) {
    const Dim &dimA = getDim(*op.getInputs()[0]);
    const Dim &dimB = getDim(*op.getInputs()[1]);
    const Dim &dimO = getDim(*op.getOutput());
    std::string lambda = "lambda ";
    for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
        lambda += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
    }
    lambda += ": I[0][";
    int axisCnt = dimB.size();
    for (size_t i = 0, iEnd = dimA.size(); i < iEnd; i++) {
        if ((int)i == op.getAxis()) {
            lambda += "I[1][(";
            for (size_t j = 0, jEnd = dimB.size(); j < jEnd; j++) {
                lambda += std::string(1, 'a' + j) + (j < jEnd - 1 ? ", " : "");
            }
            lambda += ")]";
        } else {
            lambda += std::string(1, 'a' + axisCnt++);
        }
        lambda += (i < iEnd - 1 ? ", " : "");
    }
    lambda += "]";

    std::string func = "gather_" + std::to_string(op.getGuid());
    std::string input0 = getVarName(*op.getInputs()[0]);
    std::string input1 = getVarName(*op.getInputs()[1]);
    std::string output = getVarName(*op.getOutputs()[0]);

    auto res = getTVMCode({dimA, dimB}, {"float32", "int32"}, dimO, lambda,
                          func, {input0, input1}, output);

    head += "\n" + res.first + "\n";
    main += "\n" + res.second + "\n";
}

void CodeEngine::genSplitDesc(const SplitOp &op) {
    // Empty
}

void CodeEngine::genSplitCompute(const SplitOp &op) {
    if (op.getDim() == 0) {
        return;
    }

    int offset = 0;
    const Dim &dimA = getDim(*op.getInputs()[0]);
    const int axis = op.getDim();
    for (size_t outId = 0, outNum = op.getOutputs().size(); outId < outNum;
         outId++) {
        auto &&out = op.getOutputs()[outId];
        const Dim &dimO = getDim(*out);
        std::string lambda = "lambda ";
        for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
            lambda += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
        }
        lambda += ": I[0][";
        for (size_t i = 0, iEnd = dimA.size(); i < iEnd; i++) {
            if ((int)i == axis) {
                lambda += std::string(1, 'a' + i) + " + " +
                          std::to_string(offset) + (i < iEnd - 1 ? ", " : "");
            } else {
                lambda += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
            }
        }
        lambda += "]";

        std::string func = "split_" + std::to_string(op.getGuid()) + "_" +
                           std::to_string(outId);
        std::string input = getVarName(*op.getInputs()[0]);
        std::string output = getVarName(*out);

        auto res = getTVMCode({dimA}, {"float32"}, dimO, lambda, func, {input},
                              output);

        head += "\n" + res.first + "\n";
        main += "\n" + res.second + "\n";

        offset += dimO[axis];
    }
}

void CodeEngine::genConcatDesc(const ConcatOp &op) {
    // Empty
}

void CodeEngine::genConcatCompute(const ConcatOp &op) {
    if (op.getDim() == 0) {
        return;
    }

    const Dim &dimO = getDim(*op.getOutput());
    auto axis = op.getDim();
    std::string lambda = "lambda ";
    for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
        lambda += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
    }
    lambda += ": ";
    std::function<std::string(int, int)> f = [&](int inId,
                                                 int offset) -> std::string {
        std::string str;
        auto &&in = op.getInputs()[inId];
        const Dim &dimA = getDim(*in);
        if (inId < (int)op.getInputs().size() - 1) {
            str += "tvm.tir.if_then_else(";
            str += std::string(1, 'a' + axis) + " < " +
                   std::to_string(offset + dimA[axis]) + ", ";
            str += "I[" + std::to_string(inId) + "][";
            for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
                if ((int)i == axis) {
                    str += std::string(1, 'a' + i) + " - " +
                           std::to_string(offset) + (i < iEnd - 1 ? ", " : "");
                } else {
                    str += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
                }
            }
            str += "], ";
            str += f(inId + 1, offset + dimA[axis]);
            str += ")";
        } else {
            str += "I[" + std::to_string(inId) + "][";
            for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
                if ((int)i == axis) {
                    str += std::string(1, 'a' + i) + " - " +
                           std::to_string(offset) + (i < iEnd - 1 ? ", " : "");
                } else {
                    str += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
                }
            }
            str += "]";
        }
        return str;
    };
    lambda += f(0, 0);

    std::vector<Dim> inDims;
    std::vector<std::string> inDTypes, inNames;
    for (auto &&in : op.getInputs()) {
        inDims.emplace_back(getDim(*in));
        inDTypes.emplace_back("float32");
        inNames.emplace_back(getVarName(*in));
    }
    std::string func = "concat_" + std::to_string(op.getGuid());
    std::string output = getVarName(*op.getOutputs()[0]);

    auto res =
        getTVMCode(inDims, inDTypes, dimO, lambda, func, inNames, output);

    head += "\n" + res.first + "\n";
    main += "\n" + res.second + "\n";
}

void CodeEngine::genExtendDesc(const ExtendOp &op) {
    // Empty
}

void CodeEngine::genExtendCompute(const ExtendOp &op) {
    const Dim &dimA = getDim(*op.getInputs()[0]);
    const Dim &dimO = getDim(*op.getOutput());
    auto axis = op.getDim();
    auto nCopy = op.getNum() + 1;
    std::string lambda = "lambda ";
    for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
        lambda += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
    }
    lambda += ": ";
    std::function<std::string(int, int)> f = [&](int inId,
                                                 int offset) -> std::string {
        std::string str;
        if (inId < nCopy - 1) {
            str += "tvm.tir.if_then_else(";
            str += std::string(1, 'a' + axis) + " < " +
                   std::to_string(offset + dimA[axis]) + ", ";
            str += "I[0][";
            for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
                if ((int)i == axis) {
                    str += std::string(1, 'a' + i) + " - " +
                           std::to_string(offset) + (i < iEnd - 1 ? ", " : "");
                } else {
                    str += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
                }
            }
            str += "], ";
            str += f(inId + 1, offset + dimA[axis]);
            str += ")";
        } else {
            str += "I[0][";
            for (size_t i = 0, iEnd = dimO.size(); i < iEnd; i++) {
                if ((int)i == axis) {
                    str += std::string(1, 'a' + i) + " - " +
                           std::to_string(offset) + (i < iEnd - 1 ? ", " : "");
                } else {
                    str += std::string(1, 'a' + i) + (i < iEnd - 1 ? ", " : "");
                }
            }
            str += "]";
        }
        return str;
    };
    lambda += f(0, 0);

    std::string func = "extend_" + std::to_string(op.getGuid());
    std::string input = getVarName(*op.getInputs()[0]);
    std::string output = getVarName(*op.getOutputs()[0]);

    auto res =
        getTVMCode({dimA}, {"float32"}, dimO, lambda, func, {input}, output);

    head += "\n" + res.first + "\n";
    main += "\n" + res.second + "\n";
}

void CodeEngine::genReshapeDesc(const ReshapeOp &op) {
    // Empty
}

void CodeEngine::genReshapeCompute(const ReshapeOp &op) {
    /*std::string line = "checkCudaError(cudaMemcpyAsync(";
    line += getVarName(*op.getOutputs()[0]) + ", ";
    line += getVarName(*op.getInputs()[0]) + ", ";
    line += std::to_string(getTensorSize(*op.getInputs()[0])) + ", ";
    line += "cudaMemcpyDefault));";
    emit(line);*/
    emit(fmt::format("{} = {};", getVarName(*op.getOutput()),
                     getVarName(*op.getInputs()[0])));
}

void CodeEngine::genSoftmaxDesc(const SoftmaxOp &op) {
    // Empty
    /*auto dim = getDim(*op.getInputs()[0]);
    assert(op.getAxis() == (int)dim.size() - 1);
    emit("cudnnTensorDescriptor_t " + getDescName(op) + "_in_desc;");
    emit("checkCudnnError(cudnnCreateTensorDescriptor(&" + getDescName(op) +
         "_in_desc));");
    std::string line = "checkCudnnError(cudnnSetTensor4dDescriptor(";
    line += getDescName(op) + "_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ";
    line +=
        std::to_string(getTensorNElem(*op.getInputs()[0]) / dim.back()) + ", ";
    line += std::to_string(dim.back()) + ", 1, 1));";
    emit(line);*/
}

void CodeEngine::genSoftmaxCompute(const SoftmaxOp &op) {
    emit("{");
    shiftTab(1);
    std::string x = getVarName(*op.getInputs()[0]);  // input
    std::string y = getVarName(*op.getOutputs()[0]); // output
    auto _dims = (*op.getInputs()[0]).getDims();
    int batch_size = 1, V;
    for (size_t i = 0, iEnd = _dims.size(); i < iEnd - 1; ++i)
        batch_size *= _dims[i];
    V = _dims[_dims.size() - 1];
    emit(fmt::format("int batch_size = {}, V = {};", batch_size, V));
    emit(fmt::format("int max_threadblock_size = {};", V / 8));
    emit("if (max_threadblock_size >= 256)");
    emit(fmt::format("    online_softmax<256><<<batch_size,256>>>({}, {}, V);",
                     x, y));
    emit("else if (max_threadblock_size >= 128)");
    emit(fmt::format("    online_softmax<128><<<batch_size,128>>>({}, {}, V);",
                     x, y));
    emit("else if (max_threadblock_size >= 64)");
    emit(fmt::format("    online_softmax<64><<<batch_size,64>>>({}, {}, V);", x,
                     y));
    emit("else");
    emit(fmt::format("    online_softmax<32><<<batch_size,32>>>({}, {}, V);", x,
                     y));
    shiftTab(-1);
    emit("}");
    // This code causes runtime error
    // std::string alpha = "alpha_" + std::to_string(op.getGuid());
    // std::string beta = "beta_" + std::to_string(op.getGuid());
    // emit("float " + alpha + " = 1.0f, " + beta + " = 0.0f;");
    // std::string line = "";
    // line += "checkCudnnError(cudnnSoftmaxForward(cudnn, ";
    // line += "CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, ";
    // line += "&" + alpha + ", ";
    // line += getDescName(op) + "_in_desc, ";
    // line += getVarName(*op.getInputs()[0]) + ", ";
    // line += "&" + beta + ", ";
    // line += getTensorDescName(*op.getOutputs()[0]) + ", ";
    // line += getVarName(*op.getOutputs()[0]) + "));";
    // emit(line);
}

void CodeEngine::genMemBoundDesc(const MemBoundOp &op) {
    // Empty
}

void CodeEngine::genMemBoundCompute(const MemBoundOp &op) {
    static bool checkEnvVar = false, tuning = false;
    // check env flag to enable tuning
    std::string func = "mem_bound_" + std::to_string(op.getGuid());

    if (op.isComputeWeight()) {
        emit(fmt::format(
            "\n/* {} codegen is omitted since isComputeWeight */\n", func));
        return;
    }
    if (!checkEnvVar) {
        checkEnvVar = true;
        if (auto flag = getenv("NNET_Tuning_MemBound")) {
            if (flag[0] == '1')
                tuning = true;
        }
    }
    if (!tuning) {
        emit(fmt::format(
            "\n/* {} codegen is omitted since NNET_Tuning_MemBound = 0 */\n",
            func));
        return;
    }
    // Normal membound TVM
    if (op.getHint().empty()) {
        nnet::AsTVMVisitor visitor;
        visitor.dispatch(op.getExpr());
        auto &&stmts = visitor.getStmts();
        auto &&inShapes = visitor.getInputShapes();
        auto &&outShape = visitor.getOutputShape();

        std::vector<std::string> inputs;
        for (auto &&in : op.getInputs()) {
            inputs.emplace_back(getVarName(*in));
        }
        std::string output = getVarName(*op.getOutput());

        auto res = getAnsorCode(
            inShapes, std::vector<std::string>(inShapes.size(), "float32"),
            outShape, "float32", stmts, func, inputs, output);

        head += "\n" + res.first + "\n" + "/* " + op.getExpr()->toReadable() +
                " */\n";
        main += "\n" + res.second + "\n";
    } else if (op.getHint() == "Reduce_conv3x3+1x1") {
        genReduce_merge_conv_3x3_1x1(op);
    } else
        assert(false);
}

void CodeEngine::genReduce_merge_conv_3x3_1x1(const MemBoundOp &op) {
    const auto &[n, f, h, w] = op.getNFHW();
    emit(fmt::format("hetConvToMMReduce({}, {}, {}, {}, {}, {}, {});", n, h, w,
                     f, getVarName(*op.getInputs()[0]),
                     getVarName(*op.getOutputs()[0]),
                     getVarName(*op.getInputs()[1])));
    //  var_157, var_11, var_2));
}

void CodeEngine::genG2BMMDesc(const G2BMMOp &op) {
    // Empty
}

void CodeEngine::genG2BMMCompute(const G2BMMOp &op) {
    auto A = op.getInputs()[0];
    auto &&dimA = getDim(*A);
    auto b = dimA[0];
    auto n = dimA[1];
    auto m = dimA[2];
    auto w = op.getWidth();
    auto d = op.getDilation();
    emit(fmt::format("tpm::sg2bmm({}, {}, {}, {}, {}, {}, {}, {});",
                     getVarName(*op.getInputs()[0]),
                     getVarName(*op.getInputs()[1]),
                     getVarName(*op.getOutputs()[0]), b, n, m, w, d));
}

void CodeEngine::genGBMMLDesc(const GBMMLOp &op) {
    // Empty
}

void CodeEngine::genGBMMLCompute(const GBMMLOp &op) {
    auto A = op.getInputs()[0], B = op.getInputs()[1];
    auto &&dimA = getDim(*A), &&dimB = getDim(*B);
    auto b = dimA[0];
    auto n = dimA[1];
    auto w = (dimA[2] - 1) / 2;
    auto m = dimB[2];
    auto d = op.getDilation();
    emit(fmt::format("tpm::sgbmml({}, {}, {}, {}, {}, {}, {}, {});",
                     getVarName(*op.getInputs()[0]),
                     getVarName(*op.getInputs()[1]),
                     getVarName(*op.getOutputs()[0]), b, n, m, w, d));
}

void CodeEngine::genBatchNormDesc(const BatchNormOp &op) { return; }

void CodeEngine::genBatchNormCompute(const BatchNormOp &op) { return; }

std::pair<std::string, std::string> CodeEngine::getTVMCode(
    const std::vector<std::vector<int>> &inDims,
    const std::vector<std::string> &inDTypes, const std::vector<int> &outDims,
    const std::string &lambda, const std::string &funcName,
    const std::vector<std::string> &inputNames, const std::string &outputName) {
    std::string funcCode, invokeCode;
    try {
        infini::start_interpreter();
        auto func = py::module::import("cpp_plugin").attr("gen_simple_op");
        py::tuple code = func(inDims, inDTypes, outDims, lambda, funcName,
                              inputNames, outputName);
        funcCode = py::str(code[0]), invokeCode = py::str(code[1]);
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_ImportError)) {
            std::cerr << "Import Error. Don't forget to set environment "
                         "variable PYTHONPATH to contain "
                         "<repo-root>/python"
                      << std::endl;
        }
        throw;
    }
    return std::make_pair(funcCode, invokeCode);
}

std::pair<std::string, std::string> CodeEngine::getAnsorCode(
    const std::vector<std::vector<int>> &inDims,
    const std::vector<std::string> &inDTypes, const std::vector<int> &outDims,
    const std::string &outDType, const std::string &lambda,
    const std::string &funcName, const std::vector<std::string> &inputNames,
    const std::string &outputName) {
    std::string funcCode, invokeCode;
    try {
        infini::start_interpreter();
        auto func = py::module::import("cpp_plugin").attr("gen_ansor_op");
        py::tuple code = func(inDims, inDTypes, outDims, outDType, lambda,
                              funcName, inputNames, outputName);
        funcCode = py::str(code[0]), invokeCode = py::str(code[1]);
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_ImportError)) {
            std::cerr << "Import Error. Don't forget to set environment "
                         "variable PYTHONPATH to contain "
                         "<repo-root>/python"
                      << std::endl;
        }
        throw;
    }
    return std::make_pair(funcCode, invokeCode);
}

void CodeEngine::importPerfEngine(std::shared_ptr<PerfEngine> perfEngine_) {
    perfEngine = perfEngine_;
}

} // namespace tpm
