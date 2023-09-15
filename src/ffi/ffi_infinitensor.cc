#include "common/error_handler.h"
#include "computation/graph.h"
#include "onnx/operators.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
using namespace refactor;
using namespace computation;
using Name = std::string;

class Handler {
    Graph _g;

  public:
    explicit Handler(Graph g) : _g(std::move(g)) {}
    std::unordered_set<Name> fillEdgeInfo() { return _g.fillEdgeInfo(); }
    void setInput(size_t index, std::shared_ptr<Tensor> tensor) {
        ASSERT(_g.setInput(index, std::move(tensor)),
               fmt::format("set input {} failed", index));
    }
    void substitute(const char *name, int64_t value) {
        ASSERT(_g.substitute(name, value),
               fmt::format("Variable {} not exist", name));
    }

    void runCuda() { TODO("Not implemented"); }
};

std::shared_ptr<Tensor> edge(int dataType, std::vector<DimExpr> shape,
                             std::optional<py::array> data) {
    auto ans = Tensor::share(static_cast<common::DataType>(dataType),
                             Shape(shape.begin(), shape.end()));
    if (data) {
        auto const bytesSize = ans->bytesSize();
        ASSERT(bytesSize == static_cast<size_t>(data->nbytes()),
               "Data size mismatch");
        ans->data = std::make_shared<Blob>(new uint8_t[bytesSize]);
        std::memcpy(ans->data->ptr, data->data(), bytesSize);
    }
    return ans;
}

std::shared_ptr<Operator>
node(const char *opType,
     std::unordered_map<std::string, decltype(Attribute::value)> attrs) {
    std::unordered_map<std::string, Attribute> attrs_;
    for (auto it = attrs.begin(); it != attrs.end(); attrs.erase(it++)) {
        attrs_.insert({std::move(it->first), {std::move(it->second)}});
    }
    return std::make_shared<Operator>(Operator{
        OpType::parse(fmt::format("onnx::{}", opType)), std::move(attrs_)});
}

std::shared_ptr<Handler>
graph(std::unordered_map<Name, std::pair<std::vector<Name>, std::vector<Name>>>
          topology,
      std::unordered_map<Name, std::shared_ptr<Operator>> nodes,
      std::unordered_map<Name, std::shared_ptr<Tensor>> edges,
      std::vector<Name> inputs, std::vector<Name> outputs) {
    auto builder = graph_topo::Builder<Name, Node, Name, Edge>{};
    for (auto &[node, rels] : topology) {
        auto &[inputs, outputs] = rels;
        for (auto const &input : inputs) {
            builder.edges.insert({input, {nullptr, input}});
        }
        for (auto const &output : outputs) {
            builder.edges.insert({output, {nullptr, output}});
        }
        builder.topology.insert(
            {std::move(node), {std::move(inputs), std::move(outputs)}});
    }
    builder.nodes.reserve(nodes.size());
    for (auto &[name, operator_] : nodes) {
        auto node = Node{std::move(operator_), name};
        builder.nodes.insert({std::move(name), std::move(node)});
    }
    for (auto &[name, tensor] : edges) {
        auto edge = Edge{std::move(tensor), name};
        auto it = builder.edges.find(name);
        ASSERT(it != builder.edges.end(), "Edge not connected");
        it->second.tensor = std::move(edge.tensor);
    }
    builder.globalInputs = std::move(inputs);
    builder.globalOutputs = std::move(outputs);
    return std::make_shared<Handler>(Graph(builder.build()));
}

void register_refactor(py::module &m) {
    onnx::register_();

    py::class_<DimExpr>(m, "DimExpr")
        .def(py::init<int64_t>())
        .def(py::init<std::string>());
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator");
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor");
    py::class_<Handler, std::shared_ptr<Handler>>(m, "Graph")
        .def("fill_edge_info", &Handler::fillEdgeInfo)
        .def("substitute", &Handler::substitute)
        .def("set_input", &Handler::setInput)
        .def("run_cuda", &Handler::runCuda);
    m.def("refactor_tensor", edge)
        .def("refactor_operator", node)
        .def("refactor_graph", graph);
}
} // namespace

PYBIND11_MODULE(backend, m) { register_refactor(m); }
