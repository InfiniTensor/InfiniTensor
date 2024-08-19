#include "code_gen/perf_engine.h"
#include <ctime>

// Tuple output for dumping operator args
namespace aux {
template <std::size_t...> struct seq {};

template <std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <std::size_t... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch, Tr> &os, Tuple const &t, seq<Is...>) {
    using swallow = int[];
    (void)swallow{0,
                  (void(os << (Is == 0 ? "" : ", ") << std::get<Is>(t)), 0)...};
}
} // namespace aux

template <class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr> &os, std::tuple<Args...> const &t)
    -> std::basic_ostream<Ch, Tr> & {
    os << "(";
    aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
    return os << ")";
}

namespace tpm {

void PerfEngine::allocMem() {
    // the number of elements in float type
    // (1 << 28) * sizeof(float) = 1 GB
    size_t elemNum = (1 << 28);
    // 10GB for Longformer
    // size_t longformerNum = 3lu * (1 << 30);
    size_t wsSize = 7ll << 30; // 7 GB
    checkCudaError(cudaMalloc(&inputPtr, elemNum * sizeof(float)));
    checkCudaError(cudaMalloc(&weightPtr, elemNum * sizeof(float)));
    checkCudaError(cudaMalloc(&biasPtr, elemNum * sizeof(float)));
    checkCudaError(cudaMalloc(&outputPtr, elemNum * sizeof(float)));
    checkCudaError(cudaMalloc(&workspace, wsSize));

    // reuse memory allocated for ConvOp
    matA = inputPtr;
    matB = weightPtr;
    matC = outputPtr;

    curandGenerator_t gen;
    checkCurandError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurandError(
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock()));
    checkCurandError(curandGenerateUniform(gen, inputPtr, elemNum));
    checkCurandError(curandGenerateUniform(gen, weightPtr, elemNum));
    checkCurandError(curandGenerateUniform(gen, outputPtr, elemNum));
}

void PerfEngine::dumpPerfData() {
    printf("\n============ Conv perf ============\n");
    for (const auto &kv : this->convPerf) {
        std::cout << kv.first << " : " << kv.second.time << std::endl;
    }
    printf("\n============ gemm perf ============\n");
    for (const auto &kv : this->matmulPerf) {
        std::cout << kv.first << " : " << kv.second.time << std::endl;
    }
    printf("\n============ maxpool perf ============\n");
    for (const auto &kv : this->maxPoolPerf) {
        std::cout << kv.first << " : " << kv.second << std::endl;
    }
    printf("\n============ avgpool perf ============\n");
    for (const auto &kv : this->avgPoolPerf) {
        std::cout << kv.first << " : " << kv.second << std::endl;
    }
    printf("\n============ end perf ============\n");
}

} // namespace tpm
