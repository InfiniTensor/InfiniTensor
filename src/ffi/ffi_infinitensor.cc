#include "common/error_handler.h"
#include "computation/graph.h"
#include "onnx/operators.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
using namespace refactor;
using namespace computation;

class Handler {
    Graph _g;

  public:
    explicit Handler(Graph &&g) : _g(std::forward<Graph>(g)) {}

    std::vector<std::string> fillEdgeInfo() {
        std::vector<std::string> ans;
        auto variables = _g.fillEdgeInfo();
        std::transform(variables.begin(), variables.end(),
                       std::back_inserter(ans),
                       [](auto &&v) { return std::move(v); });
        return ans;
    }
};

Edge edge(int dataType, std::vector<DimExpr> shape,
          std::optional<std::vector<uint8_t>> data) {
    Shape s(shape.begin(), shape.end());
    auto ans = std::make_shared<Tensor>(static_cast<common::DataType>(dataType),
                                        std::move(s));
    if (data) {
        auto const bytesSize = ans->bytesSize();
        ASSERT(bytesSize == data->size(), "Data size mismatch");
        ans->data = std::make_shared<Blob>(new uint8_t[bytesSize]);
        std::memcpy(ans->data->ptr, data->data(), bytesSize);
    }
    return ans;
}

Node node(std::string opType,
          std::unordered_map<std::string, decltype(Attribute::value)> attrs) {
    std::unordered_map<std::string, Attribute> attrs_;
    for (auto it = attrs.begin(); it != attrs.end(); attrs.erase(it++)) {
        attrs_.insert({std::move(it->first), {std::move(it->second)}});
    }
    return std::make_shared<Operator>(
        Operator{OpType::parse(fmt::format("onnx::{}", opType).c_str()),
                 std::move(attrs_)});
}

std::shared_ptr<Handler>
graph(std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                                std::vector<std::string>>>
          topology,
      std::unordered_map<std::string, Node> nodes,
      std::unordered_map<std::string, Edge> edges,
      std::vector<std::string> inputs, std::vector<std::string> outputs) {
    using Name = std::string;
    auto builder = graph_topo::Builder<Name, Node, Name, Edge>{};
    for (auto &[node, rels] : topology) {
        builder.topology.insert(
            {std::move(node), {std::move(rels.first), std::move(rels.second)}});
    }
    builder.nodes = std::move(nodes);
    builder.edges = std::move(edges);
    builder.globalInputs = std::move(inputs);
    builder.globalOutputs = std::move(outputs);
    return std::make_shared<Handler>(Graph(builder.build()));
}

void register_refactor(py::module &m) {
    onnx::register_();

    py::class_<DimExpr>(m, "DimExpr")
        .def(py::init<int64_t>())
        .def(py::init<std::string &&>());
    py::class_<Operator, Node>(m, "Node");
    py::class_<Tensor, Edge>(m, "Edge");
    py::class_<Handler, std::shared_ptr<Handler>>(m, "Graph")
        .def("fill_edge_info", &Handler::fillEdgeInfo);
    m.def("refactor_tensor", edge)
        .def("refactor_operator", node)
        .def("refactor_graph", graph);
}
} // namespace

PYBIND11_MODULE(backend, m) { register_refactor(m); }
