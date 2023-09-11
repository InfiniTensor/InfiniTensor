#include "../RefactorGraph/src/02computation/include/graph/graph.h"
#include "common/error_handler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
using namespace refactor;

class Handler {
    graph::Graph _g;

  public:
    explicit Handler(graph::Graph &&g) : _g(std::forward<graph::Graph>(g)) {}
};

graph::Edge edge(int dataType, std::vector<graph::DimExpr> shape,
                 std::optional<std::vector<uint8_t>> data) {
    graph::Shape s(shape.begin(), shape.end());
    auto ans = std::make_shared<graph::Tensor>(
        static_cast<common::DataType>(dataType), std::move(s));
    if (data) {
        auto const bytesSize = ans->bytesSize();
        ASSERT(bytesSize == data->size(), "Data size mismatch");
        ans->data = std::make_shared<graph::Blob>(new uint8_t[bytesSize]);
        std::memcpy(ans->data->ptr, data->data(), bytesSize);
    }
    return ans;
}

graph::Node
node(std::string opType,
     std::unordered_map<std::string, decltype(graph::Attribute::value)> attrs) {
    std::unordered_map<std::string, graph::Attribute> attrs_;
    for (auto it = attrs.begin(); it != attrs.end(); attrs.erase(it++)) {
        attrs_.insert({std::move(it->first), {std::move(it->second)}});
    }
    return std::make_shared<graph::NodeInfo>(
        graph::Operator{common::OpType::Unknown, std::move(attrs_)});
}

std::shared_ptr<Handler>
graph(std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                                std::vector<std::string>>>
          topology,
      std::unordered_map<std::string, graph::Node> nodes,
      std::unordered_map<std::string, graph::Edge> edges,
      std::vector<std::string> inputs, std::vector<std::string> outputs) {
    using Name = std::string;
    auto builder = graph_topo::Builder<Name, graph::Node, Name, graph::Edge>{};
    for (auto &[node, rels] : topology) {
        builder.topology.insert(
            {std::move(node), {std::move(rels.first), std::move(rels.second)}});
    }
    builder.nodes = std::move(nodes);
    builder.edges = std::move(edges);
    builder.globalInputs = std::move(inputs);
    builder.globalOutputs = std::move(outputs);
    return std::make_shared<Handler>(graph::Graph(builder.build()));
}

void register_refactor(py::module &m) {
    py::class_<graph::DimExpr>(m, "DimExpr")
        .def(py::init<int64_t>())
        .def(py::init<std::string &&>());
    py::class_<graph::NodeInfo, graph::Node>(m, "Node");
    py::class_<graph::Tensor, graph::Edge>(m, "Edge");
    py::class_<Handler, std ::shared_ptr<Handler>>(m, "Graph");
    m.def("refactor_tensor", edge)
        .def("refactor_operator", node)
        .def("refactor_graph", graph);
}
} // namespace

PYBIND11_MODULE(backend, m) { register_refactor(m); }
