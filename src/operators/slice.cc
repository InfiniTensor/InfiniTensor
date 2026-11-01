#include "operators/slice.h"

namespace infini {
SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &starts, const vector<int> &ends,
                   const optional<vector<int>> &_axes,
                   const optional<vector<int>> &_steps)
    : OperatorObj(OpType::Slice, {input}, {output}) {
    setSliceParameters(starts, ends, _axes, _steps);
    IT_ASSERT(checkValid(graph));
}

void SliceObj::setSliceParameters(const vector<int> &starts,
                                  const vector<int> &ends,
                                  const optional<vector<int>> &_axes,
                                  const optional<vector<int>> &_steps) {
    auto previous = axes;
    try {
        buildAxes(inputs[0]->getDims(), starts, ends, _axes, _steps);
    } catch (...) {
        axes = std::move(previous);
        throw;
    }
    rawStarts = starts;
    rawEnds = ends;
    rawAxes = _axes;
    rawSteps = _steps;
}

void SliceObj::buildAxes(const Shape &shape, const vector<int> &starts,
                         const vector<int> &ends,
                         const optional<vector<int>> &_axes,
                         const optional<vector<int>> &_steps) {
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
                IT_ASSERT(index >= 0 && index < static_cast<int>(shape.size()),
                          "Slice axis out of range");
                IT_ASSERT(!axes.count(index), "Repeated Slice axis");
                axes[index] = i;
            }
        } else
            for (size_t i = 0; i < size; ++i) {
                IT_ASSERT(i < shape.size(), "Slice axis out of range");
                axes[i] = i;
            }

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

    this->axes.clear();
    auto size = shape.size();
    this->axes.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        auto len = shape[i];
        if (auto _i = axes.find(i); _i != axes.end()) {
            auto __i = _i->second;
            auto start = starts[__i];
            auto end = ends[__i];
            if (start > len)
                start = len;
            if (end > len)
                end = len;
            IT_ASSERT(steps[__i] > 0,
                      "Data Slice currently requires positive steps");
            start = std::clamp(start >= 0 ? start : start + len, 0, len);
            end = std::clamp(end >= 0 ? end : end + len, 0, len);
            this->axes.push_back({start, std::max(start, end), steps[__i]});
        } else {
            this->axes.push_back({0, len, 1});
        }
    }
}

optional<vector<Shape>> SliceObj::inferShape(const TensorVec &inputs) {
    // Even constant endpoints must be normalized against the current input.
    buildAxes(inputs[0]->getDims(), rawStarts, rawEnds, rawAxes, rawSteps);
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
    vector<int> ans{type.underlying()};
    for (const auto &range : axes) {
        ans.push_back(range.start);
        ans.push_back(range.end);
        ans.push_back(range.step);
    }
    return ans;
}

} // namespace infini
