#include "operators/squeeze.h"
#include "utils/operator_utils.h"
#include <algorithm>

namespace infini {
SqueezeObj::SqueezeObj(GraphObj *graph, Tensor input, Tensor output, Shape axes)
    : OperatorObj(OpType::Squeeze, {input}, {output}), axes(std::move(axes)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SqueezeObj::inferShape(const TensorVec &inputs) {
    Shape inputDim = inputs[0]->getDims();
    Shape outputShape;
    auto rank = inputs[0]->getRank();
    if (axes.size() == 0) {
        for (int i = 0; i < (int)rank; ++i) {
            if (inputDim[i] == 1) {
                axes.emplace_back(i);
            }
        }
    }
    auto new_axes = axes;
    std::transform(axes.begin(), axes.end(), new_axes.begin(),
                   [inputDim, rank](auto x) {
                       x = get_real_axis(x, rank);
                       IT_ASSERT(inputDim[x] == 1);
                       return x;
                   });
    for (int i = 0; i < (int)rank; ++i) {
        auto it = std::find(new_axes.begin(), new_axes.end(), i);
        if (it == new_axes.end()) {
            outputShape.emplace_back(inputDim[i]);
        }
    }
    return {{outputShape}};
}

void SqueezeObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    // Dropping dimensions of length one leaves the elements themselves alone.
    const auto &value = *inputs[0]->getShapeValue();
    if (value.size() != outputs[0]->size()) {
        return;
    }
    vector<bool> fixed;
    fixed.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        fixed.push_back(inputs[0]->isShapeValueFixed(i));
    }
    outputs[0]->setShapeValue(value, std::move(fixed));
}

std::string SqueezeObj::toString() const {
    std::ostringstream os;
    os << "Squeeze[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "axes=" << vecToString(axes) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SqueezeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), axes.begin(), axes.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> SqueezeObj::getOpAttrVector() const {
    vector<int> ret = axes;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

} // namespace infini
