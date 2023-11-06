#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
    auto rank = input->getRank();
    if (permute.empty()) {
        for (size_t i = 0; i < rank; ++i) {
            transposePermute[i] = i;
        }
    } else {
        IT_ASSERT(rank == permute.size());
        transposePermute = std::move(permute);
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
TransposeObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    int rank = A->getRank();

    for (auto index : transposePermute) {
        IT_ASSERT(index < rank);
    }
    for (int i = 0; i < rank; ++i) {
        output_dim[i] = input_dim[transposePermute[i]];
    }
    return {{output_dim}};
}

std::string TransposeObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> TransposeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> TransposeObj::getOpAttrVector() const {
    return {type.underlying()};
}

DepthToSpaceObj::DepthToSpaceObj(GraphObj *graph, Tensor input, Tensor output,
                                 int blocksize, std::string mode)
    : OperatorObj(OpType::DepthToSpace, {input}, {output}) {
    blockSize = blocksize;
    D2SMode = 0;
    D2SModeString = "DCR";
    if (mode == "CRD") {
        D2SMode = 1;
        D2SModeString = "CRD";
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
DepthToSpaceObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    auto inputDim = A->getDims();
    IT_ASSERT(inputDim.size() == 4);
    if (D2SMode == 0) {
        reshapeDim[0] = inputDim[0];
        reshapeDim[1] = blockSize;
        reshapeDim[2] = blockSize;
        reshapeDim[3] = inputDim[1] / (blockSize * blockSize);
        reshapeDim[4] = inputDim[2];
        reshapeDim[5] = inputDim[3];
        transposeDim[0] = reshapeDim[0];
        transposeDim[1] = reshapeDim[3];
        transposeDim[2] = reshapeDim[4];
        transposeDim[3] = reshapeDim[1];
        transposeDim[4] = reshapeDim[5];
        transposeDim[5] = reshapeDim[2];
        outDim[0] = inputDim[0];
        outDim[1] = inputDim[1] / (blockSize * blockSize);
        outDim[2] = inputDim[2] * blockSize;
        outDim[3] = inputDim[3] * blockSize;
    } else {
        reshapeDim[0] = inputDim[0];
        reshapeDim[1] = inputDim[1] / (blockSize * blockSize);
        reshapeDim[2] = blockSize;
        reshapeDim[3] = blockSize;
        reshapeDim[4] = inputDim[2];
        reshapeDim[5] = inputDim[3];
        transposeDim[0] = reshapeDim[0];
        transposeDim[1] = reshapeDim[1];
        transposeDim[2] = reshapeDim[4];
        transposeDim[3] = reshapeDim[2];
        transposeDim[4] = reshapeDim[5];
        transposeDim[5] = reshapeDim[3];
        outDim[0] = inputDim[0];
        outDim[1] = inputDim[1] / (blockSize * blockSize);
        outDim[2] = inputDim[2] * blockSize;
        outDim[3] = inputDim[3] * blockSize;
    }

    return {{outDim}};
}

std::string DepthToSpaceObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> DepthToSpaceObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> DepthToSpaceObj::getOpAttrVector() const {
    return {type.underlying()};
}

}; // namespace infini
