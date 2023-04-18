#include "operators/slice.h"

namespace infini {
SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &starts, const vector<int> &ends,
                   const optional<vector<int>> &_axes,
                   const optional<vector<int>> &_steps)
    : OperatorObj(OpType::Slice, {input}, {output}) {
    auto shape = input->getDims(); // shape of input
    map<size_t, size_t> axes;
    vector<int> steps;
    {
        auto size = starts.size();      // size of starts
        IT_ASSERT(size == ends.size()); // size of ends

        if (_axes) {
            IT_ASSERT(size == _axes->size());
            // onnx doc: "Behavior is undefined if an axis is repeated."
            IT_ASSERT(size == std::set(_axes->begin(), _axes->end()).size());

            for (size_t i = 0; i < size; ++i) {
                auto index = _axes->at(i);
                if (index < 0)
                    index += shape.size();
                axes[index] = i;
            }
        } else
            for (size_t i = 0; i < size; ++i)
                axes[i] = i;

        if (_steps) {
            IT_ASSERT(size == _steps->size());
            // onnx doc: "‘steps’ cannot be 0."
            IT_ASSERT(std::find(_steps->begin(), _steps->end(), 0) ==
                      _steps->end());
            steps = *_steps;
        } else {
            steps.reserve(size);
            for (size_t i = 0; i < size; ++i)
                steps.push_back(1);
        }
    }

    auto size = shape.size();
    this->axes.reserve(size);
    for (size_t i = 0; i < size; ++i)
        if (auto _i = axes.find(i); _i != axes.end()) {
            auto __i = _i->second;
            auto start = starts[__i];
            auto end = ends[__i];
            this->axes.push_back({start >= 0 ? start : start + shape[__i],
                                  end >= 0 ? end : end + shape[__i],
                                  steps[__i]});
        } else {
            this->axes.push_back({0, shape[i], 1});
        }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SliceObj::inferShape(const TensorVec &inputs) const {
    Shape ans;
    ans.reserve(axes.size());
    for (const auto &range : axes) {
        auto step = std::abs(range.step);
        ans.push_back((range.end - range.start + step - 1) / step);
    }
    return {{ans}};
}

std::string SliceObj::toString() const {
    std::ostringstream os;
    os << "Slice[" << getGuid() << "][";
    for (const auto &range : axes) {
        os << range.start << ':' << range.step << ':' << range.end << ", ";
    }
    os << "]("
       << "input=" << inputs[0]->getGuid() << ", "
       << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SliceObj::getWorkloadVector() const {
    auto ans = getOpAttrVector();
    {
        auto i = inputs[0]->getDims();
        ans.insert(ans.end(), i.begin(), i.end());
    }
    if (!outputs.empty()) {
        auto o = outputs[0]->getDims();
        ans.insert(ans.end(), o.begin(), o.end());
    }
    return ans;
}

vector<int> SliceObj::getOpAttrVector() const {
    vector<int> ans{enum_to_underlying(type)};
    for (const auto &range : axes) {
        ans.push_back(range.start);
        ans.push_back(range.end);
        ans.push_back(range.step);
    }
    return ans;
}

} // namespace infini
