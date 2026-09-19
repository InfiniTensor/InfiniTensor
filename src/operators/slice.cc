#include "operators/slice.h"

namespace infini {
void SliceObj::resolveRanges(const Shape &shape) {
    const auto size = starts.size();
    IT_ASSERT(size == ends.size(),
              "Slice starts and ends must have equal size");
    IT_ASSERT(!namedAxes || namedAxes->size() == size);
    IT_ASSERT(namedAxes || size <= shape.size(), "too many Slice axes");
    IT_ASSERT(!namedSteps || namedSteps->size() == size);
    map<size_t, size_t> positions;
    for (size_t i = 0; i < size; ++i) {
        int64_t axis = namedAxes ? namedAxes->at(i) : static_cast<int64_t>(i);
        if (axis < 0)
            axis += static_cast<int64_t>(shape.size());
        IT_ASSERT(axis >= 0 && axis < static_cast<int64_t>(shape.size()),
                  "Slice axis is out of range");
        IT_ASSERT(positions.emplace(static_cast<size_t>(axis), i).second,
                  "Slice axes must not repeat after normalization");
        IT_ASSERT(!namedSteps || namedSteps->at(i) != 0,
                  "Slice step must not be zero");
    }

    this->axes.clear();
    this->axes.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        const int64_t len = shape[i];
        if (auto at = positions.find(i); at != positions.end()) {
            const auto position = at->second;
            const auto step = namedSteps ? namedSteps->at(position) : 1;
            // Keep ONNX's int64 sentinels intact until they have been resolved
            // against the current dimension. Only the clamped range fits int.
            auto start = starts[position];
            auto end = ends[position];
            if (start < 0)
                start += len;
            if (end < 0)
                end += len;
            if (step > 0) {
                start = std::clamp<int64_t>(start, 0, len);
                end = std::clamp<int64_t>(end, 0, len);
            } else {
                start = len == 0 ? -1 : std::clamp<int64_t>(start, 0, len - 1);
                end = std::clamp<int64_t>(end, -1, len - 1);
            }
            this->axes.push_back(
                {static_cast<int>(start), static_cast<int>(end), step});
        } else {
            // A dimension the model did not ask about is taken whole, so its
            // extent is the input's own -- which is the shape it has now, not
            // the one it had when this operator was built.
            this->axes.push_back({0, static_cast<int>(len), 1});
        }
    }
}

SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &_starts, const vector<int> &_ends,
                   const optional<vector<int>> &_axes,
                   const optional<vector<int>> &_steps)
    : OperatorObj(OpType::Slice, {input}, {output}),
      starts(_starts.begin(), _starts.end()), ends(_ends.begin(), _ends.end()),
      namedAxes(_axes), namedSteps(_steps) {
    resolveRanges(input->getDims());
    IT_ASSERT(checkValid(graph));
}

SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor starts, Tensor ends,
                   Tensor output, const optional<vector<int>> &_axes,
                   const optional<vector<int>> &_steps)
    : OperatorObj(OpType::Slice, {input, starts, ends}, {output}),
      namedAxes(_axes), namedSteps(_steps), boundsFromEdges(true) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SliceObj::inferShape(const TensorVec &inputs) {
    // The window and the extents it leaves alone both move with the input's
    // shape, so they are worked out here rather than kept from an earlier one.
    if (boundsFromEdges) {
        for (size_t i : {1, 2}) {
            IT_ASSERT(inputs[i]->getRank() == 1 &&
                          (inputs[i]->getDType() == DataType::Int32 ||
                           inputs[i]->getDType() == DataType::Int64),
                      "Slice bounds must be rank-one int32 or int64 tensors");
        }
        IT_ASSERT(inputs[1]->getShapeValue().has_value() &&
                      inputs[2]->getShapeValue().has_value(),
                  "the bounds of this Slice are not known while shapes are "
                  "inferred, so they hold data rather than positions");
        const auto &rawStarts = *inputs[1]->getShapeValue();
        const auto &rawEnds = *inputs[2]->getShapeValue();
        IT_ASSERT(rawStarts.size() == inputs[1]->size() &&
                      rawEnds.size() == inputs[2]->size(),
                  "Slice bound values must match their tensor sizes");
        starts = rawStarts;
        ends = rawEnds;
    }
    resolveRanges(inputs[0]->getDims());

    Shape ans;
    ans.reserve(axes.size());
    for (const auto &range : axes) {
        const int64_t step = range.step;
        const int64_t distance = step > 0 ? int64_t(range.end) - range.start
                                          : int64_t(range.start) - range.end;
        const auto stride = step > 0 ? step : -step;
        ans.push_back(static_cast<int>(
            (std::max<int64_t>(distance, 0) + stride - 1) / stride));
    }
    return {{ans}};
}

void SliceObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    // A shape is a list, and taking part of a list is the one case worth
    // handling: a model that reshapes to "everything but the last dimension,
    // then whatever is left" writes exactly this, and a simplifier will produce
    // it even where the model did not. Slicing anything of higher rank is a
    // slice of data rather than of dimensions, and nothing here can say what
    // the result would be.
    if (inputs[0]->getRank() != 1 || axes.size() != 1) {
        return;
    }
    const auto &value = *inputs[0]->getShapeValue();
    const auto &range = axes[0];
    // The range has been resolved against the current input. A fixed data
    // element is only a fixed result when the selection itself cannot move.
    const bool boundsFixed =
        !boundsFromEdges || (inputs[1]->isShapeValueWhollyFixed() &&
                             inputs[2]->isShapeValueWhollyFixed());
    vector<int64_t> picked;
    vector<bool> fixed;
    for (int64_t i = range.start;
         range.step > 0 ? i < range.end : i > range.end; i += range.step) {
        if (i < 0 || i >= static_cast<int64_t>(value.size())) {
            return;
        }
        picked.push_back(value[i]);
        fixed.push_back(boundsFixed && inputs[0]->isShapeValueFixed(i));
    }
    // Whatever the range worked out to has to be what the output was shaped
    // for; anything else means the two disagree and the value is not usable.
    if (picked.size() != outputs[0]->size()) {
        return;
    }
    outputs[0]->setShapeValue(std::move(picked), std::move(fixed));
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
