#include "operators/expand.h"

namespace infini{

ExpandObj::ExpandObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
				: OperatorObj(OpType::Expand, {input}, {output}), dims(std::move(dims)) {
				IT_ASSERT(checkValid(graph));				
}

optional<vector<Shape>> ExpandObj::inferShape(const TensorVec &inputs) const {
		auto shape_input = inputs[0]->getDims();
		Shape ret;
		size_t dim = shape_input.size();
		for(auto it = dims.rbegin(); it!= dims.rend(); ++it) {
				auto dim_ele = *(it);
				if(dim == 0) {
						ret.emplace_back(dim_ele);
						continue;
				}
			  if (dim_ele == shape_input[--dim] || dim_ele == 1 || shape_input[dim] == 1) {
						ret.emplace_back(std::max(dim_ele, shape_input[dim]));
				}
				else {return {};}
		}
		std::reverse(ret.begin(), ret.end());
		return {{ret}};
}

std::string ExpandObj::toString() const {
    std::ostringstream os;
    os << "Expand[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "dims=" << vecToString(dims) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ExpandObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
		ret.insert(ret.end(), dims.begin(), dims.end());
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> ExpandObj::getOpAttrVector() const {
		vector<int> ret = dims;
		ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

} // namespace infini
