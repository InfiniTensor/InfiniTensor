#include "operators/motion.h"
#include "utils/motion_compute.h"
namespace infini {
MotionObj::MotionObj(GraphObj *graph, OpType kind, TensorVec values, bool qemu)
    : OperatorObj(kind, values,
                  TensorVec(kind == OpType::MotionFeatures ? 4 : 1, nullptr)),
      useQemu(qemu) {
    motion::require(kind == OpType::MotionFeatures ||
                        kind == OpType::MotionReadout,
                    "Motion: unsupported operator");
    motion::require(checkValid(graph), "Motion: invalid shape or dtype");
}
optional<vector<Shape>> MotionObj::inferShape(const TensorVec &v) {
    if (int(v.size()) != numInputs() || v[0]->getDims().size() != 2)
        return {};
    int s = v[0]->getDims()[0];
    if (s < 1 || s > 1024)
        return {};
    vector<Shape> expected;
    if (type == OpType::MotionFeatures) {
        expected = {{s, 21},   {21},       {s, 7}, {3, 113},
                    {42, 113}, {113, 113}, {113},  {113}};
        for (auto &t : v)
            if (!(t->getDType() == DataType::Double))
                return {};
    } else {
        expected = {{s, 120}, {120, 128}, {128}, {128, 7}, {7}, {7}, {7}};
        for (int i = 0; i < 7; ++i)
            if (!(v[i]->getDType() ==
                  (i < 5 ? DataType::Float32 : DataType::Double)))
                return {};
    }
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i]->getDims() != expected[i])
            return {};
    if (type == OpType::MotionFeatures)
        return vector<Shape>{{s, 120}, {3, 113}, {21}, {s, 113}};
    return vector<Shape>{{s, 7}};
}
vector<DataType> MotionObj::inferDataType(const TensorVec &) const {
    if (type == OpType::MotionFeatures)
        return {DataType::Float32, DataType::Double, DataType::Double,
                DataType::Int32};
    return {DataType::Double};
}
vector<int> MotionObj::getOpAttrVector() const {
    return {type.underlying(), int(useQemu)};
}
vector<int> MotionObj::getWorkloadVector() const {
    auto result = getOpAttrVector();
    result.push_back(inputs[0]->getDims()[0]);
    return result;
}
string MotionObj::toString() const {
    return string(type.toString()) + vecToString(inputs[0]->getDims()) +
           (useQemu ? " qemu" : " reference");
}
} // namespace infini
