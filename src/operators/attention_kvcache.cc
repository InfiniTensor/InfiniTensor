#include "operators/attention_kvcache.h"
#include "utils/operator_utils.h"

namespace infini {
AttentionKVCacheObj::AttentionKVCacheObj(
    GraphObj *graph, Tensor input_k_cache, Tensor input_v_cache, Tensor input_q,
    Tensor input_k, Tensor input_v, Tensor position_id, Tensor output_matmul,
    Tensor output_k_cache, Tensor output_v_cache)
    : OperatorObj(OpType::AttentionKVCache,
                  TensorVec{input_k_cache, input_v_cache, input_q, input_k,
                            input_v, position_id},
                  TensorVec{output_matmul, output_k_cache, output_v_cache}) {
    int rank = inputs[0]->getRank();
    IT_ASSERT(rank == 4);
    dim = 2;
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
AttentionKVCacheObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(inputs.size() == 6);
    Shape dims = inputs[0]->getDims();
    ShapeElem n = dims.at(dim);
    dims[dim] = n + 1;
    return {{inputs[2]->getDims(), dims, dims}};
}

std::string AttentionKVCacheObj::toString() const {
    std::ostringstream os;
    os << "AttentionKVCache[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> AttentionKVCacheObj::getWorkloadVector() const {
    vector<int> ret = getOutputs()[0]->getDims();
    ret.emplace(ret.begin(), (int)inputs.size());
    ret.emplace(ret.begin(), dim);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> AttentionKVCacheObj::getOpAttrVector() const {
    return {type.underlying(), dim};
}

} // namespace infini
