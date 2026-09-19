#include "operators/tile.h"

namespace infini {

TileObj::TileObj(GraphObj *graph, Tensor input, Tensor output, Shape repeats)
    : OperatorObj(OpType::Tile, {input}, {output}),
      repeats(std::move(repeats)) {
    IT_ASSERT(checkValid(graph));
}

TileObj::TileObj(GraphObj *graph, Tensor input, Tensor repeats, Tensor output)
    : OperatorObj(OpType::Tile, {input, repeats}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> TileObj::inferShape(const TensorVec &inputs) {
    if (inputs.size() == 2) {
        IT_ASSERT(inputs[1]->getShapeValue().has_value(),
                  "the repeat counts of this Tile are not known while shapes "
                  "are inferred, so they hold data rather than dimensions");
        repeats = inputs[1]->getShapeValueAsShape();
    }
    const Shape &in = inputs[0]->getDims();
    // ONNX gives one count per input dimension, so a mismatch is not something
    // to line up but a target that does not describe this input.
    IT_ASSERT(repeats.size() == in.size(),
              "this Tile repeats " + std::to_string(repeats.size()) +
                  " dimensions, but its input has " +
                  std::to_string(in.size()));
    Shape ret(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        IT_ASSERT(repeats[i] >= 0,
                  "a Tile repeats a dimension a whole number of times, so " +
                      std::to_string(repeats[i]) + " is not a count");
        ret[i] = in[i] * repeats[i];
    }
    return {{ret}};
}

vector<DimSource> TileObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    IT_ASSERT(dim < outputs[0]->getRank());
    // A count read from an edge is a number only for the shape the graph
    // currently holds. Where it may still change, so may the multiple this
    // dimension is of its input, and neither answer below can be claimed.
    if (inputs.size() == 2 && !inputs[1]->isShapeValueFixed(dim)) {
        return OperatorObj::dimSources(output, dim);
    }
    // Repeating a dimension no times empties it, and an empty dimension is
    // empty whatever the input had there, so it follows nothing.
    if (repeats[dim] == 0) {
        return {};
    }
    // Otherwise the output is a fixed multiple of the input's own extent, so it
    // varies exactly when that extent does.
    return {DimSource{0, dim}};
}

std::string TileObj::toString() const {
    std::ostringstream os;
    os << "Tile[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "repeats=" << vecToString(repeats) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> TileObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), repeats.begin(), repeats.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> TileObj::getOpAttrVector() const {
    vector<int> ret = repeats;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

} // namespace infini
