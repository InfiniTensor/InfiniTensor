#include "operators/where.h"

namespace infini {

WhereObj::WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY,
                   Tensor condition, Tensor output)
    : OperatorObj(OpType::Where, TensorVec{inputX, inputY, condition},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> WhereObj::inferShape(const TensorVec &inputs) const {
    auto shape_X = inputs[0]->getDims();
    std::reverse(shape_X.begin(), shape_X.end());
    auto shape_Y = inputs[1]->getDims();
    std::reverse(shape_Y.begin(), shape_Y.end());
    auto shape_con = inputs[2]->getDims();
    std::reverse(shape_con.begin(), shape_con.end());

    int dim_X = shape_X.size();
    int dim_Y = shape_Y.size();
    int dim_con = shape_con.size();
    Shape ret;

    auto dims = std::max(dim_X, dim_Y);
    if (dim_X < dims) {
        for (int i = dim_X; i < dims; ++i)
            shape_X.emplace_back(1);
    }
    if (dim_Y < dims) {
        for (int i = dim_Y; i < dims; ++i)
            shape_Y.emplace_back(1);
    }
    if (dim_con < dims) {
        for (int i = dim_con; i < dims; ++i)
            shape_con.emplace_back(1);
    }

    for (int i = 0; i < dims; ++i) {
        if ((shape_X[i] == shape_Y[i] && shape_Y[i] == shape_con[i]) ||
            shape_X[i] == 1 || shape_Y[i] == 1 || shape_con[i] == 1) {
            ret.emplace_back(
                std::max(std::max(shape_X[i], shape_Y[i]), shape_con[i]));
        } else {
            return {};
        }
    }
    std::reverse(ret.begin(), ret.end());
    return {{ret}};
}

std::string WhereObj::toString() const {
    std::ostringstream os;
    os << "Where[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[2]->getDims()) << ",";
    os << "inputX=" << inputs[0]->getGuid() << ",";
    os << "inputY=" << inputs[1]->getGuid() << ",";
    os << "condition=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> WhereObj::getWorkloadVector() const {
    vector<int> ret = inputs[2]->getDims();
    return ret;
}

vector<int> WhereObj::getOpAttrVector() const {
    vector<int> ret = {};
    return ret;
}

} // namespace infini
