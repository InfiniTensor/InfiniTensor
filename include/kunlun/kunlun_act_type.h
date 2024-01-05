#include "core/op_type.h"
#include "kunlun/kunlun_common.h"

namespace infini {
using KunlunActType = xdnn::Activation_t;
KunlunActType parseActType(ActType act) {
    switch (act) {
    case ActType::None:
        return KunlunActType::LINEAR;
    case ActType::Tanh:
        return KunlunActType::TANH;
    case ActType::Sigmoid:
        return KunlunActType::SIGMOID;
    case ActType::Relu:
        return KunlunActType::RELU6;
    default:
        fprintf(stderr, "Activation Type not support yet!\n");
        break;
    }
    return KunlunActType::LINEAR;
}

}; // namespace infini