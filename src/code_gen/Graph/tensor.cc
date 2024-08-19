#include "code_gen/tensor.h"
#include "code_gen/common.h"
#include "code_gen/operator.h"

namespace tpm {
std::pair<Operator *, int> Tensor::getOutputOfWithIndex() {
    if (outputOf == nullptr)
        return {nullptr, -1};
    auto it = std::find(outputOf->getOutputs().begin(),
                        outputOf->getOutputs().end(), this);
    if (it != outputOf->getOutputs().end())
        return {outputOf, std::distance(it, outputOf->getOutputs().begin())};
    return {nullptr, -1};
}

bool Tensor::random_inited;
int Tensor::random_seed[256 * 16];

void printTensor(tpm::Tensor *tensor) {
    auto data = tensor->getDataPtr();
    auto sz = tensor->size();
    for (size_t i = 0; i < sz; ++i) {
        std::cout << data[i] << ", ";
        if (i % 14 == 13)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Tensor::printShape() {
    std::cout << "[";
    for (size_t i = 0; i < dims.size() - 1; ++i)
        std::cout << dims[i] << ", ";
    std::cout << dims[dims.size() - 1] << "]" << std::endl;
}
} // end of namespace tpm