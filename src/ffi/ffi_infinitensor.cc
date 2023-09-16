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
    auto const &graph() const { return _g.internal(); }
    void runCuda() { TODO("Not implemented"); }
};

class Iterator {
    std::shared_ptr<Handler> _internal;
    graph_topo::GraphTopo::Iterator _it;

  public:
    explicit Iterator(std::shared_ptr<Handler> internal)
        : _internal(std::move(internal)),
          _it(_internal->graph().topology.begin()) {}
    using T = std::tuple<Name, int, std::vector<std::variant<Name, int>>>;
    using O = std::tuple<Name, Name,
                         std::unordered_map<Name, decltype(Attribute::value)>,
                         std::vector<Name>, std::vector<Name>>;

    T buildT(size_t edgeIdx) const {
        auto const &edge = _internal->graph().edges[edgeIdx];
        auto const &shape = edge.tensor->shape;
        std::vector<std::variant<Name, int>> shape_(shape.size(), 1);
        std::transform(shape.begin(), shape.end(), shape_.begin(),
                       [](auto const &d) -> std::variant<Name, int> {
                           if (d.isVariable()) {
                               return d.variable()->name;
                           } else {
                               return static_cast<int>(d.value());
                           }
                       });
        return T{edge.name, static_cast<int>(edge.tensor->dataType),
                 std::move(shape_)};
    }

    std::vector<T> globalInputs() const {
        auto inputs = _it.globalInputs();
        std::vector<T> ans(inputs.size());
        std::transform(inputs.begin(), inputs.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return buildT(edgeIdx); });
        return ans;
    }

    std::vector<T> globalOutputs() const {
        auto outputs = _it.globalInputs();
        std::vector<T> ans(outputs.size());
        std::transform(outputs.begin(), outputs.end(), ans.begin(),
                       [this](auto const &edgeIdx) { return buildT(edgeIdx); });
        return ans;
    }

    std::optional<O> next() {
        if (_it == _internal->graph().topology.end()) {
            return std::nullopt;
        }
        auto [nodeIdx, inputs_, outputs_] = *_it++;
        auto const &node = _internal->graph().nodes[nodeIdx];
        auto const &name = node.name;
        auto const opType = node.op->opType.name();
        ASSERT(opType.substr(0, 6) == "onnx::", "Invalid opType");
        auto const &attributes = node.op->attributes;
        std::vector<Name> inputs(inputs_.size()), outputs(outputs_.size());
        std::transform(inputs_.begin(), inputs_.end(), inputs.begin(),
                       [this](auto const &idx) {
                           return _internal->graph().edges[idx].name;
                       });
        std::transform(outputs_.begin(), outputs_.end(), outputs.begin(),
                       [this](auto const &idx) {
                           return _internal->graph().edges[idx].name;
                       });
        std::unordered_map<Name, decltype(Attribute::value)> attributes_;
        attributes_.reserve(attributes.size());
        for (auto const &[name, attr] : attributes) {
            if (!std::holds_alternative<Tensor_>(attr.value)) {
                attributes_.insert({name, attr.value});
            }
        }
        return O{name, opType.substr(6), attributes_, inputs, outputs};
    }
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
    py::class_<Iterator>(m, "Iterator")
        .def(py::init<std::shared_ptr<Handler>>())
        .def("global_inputs", &Iterator::globalInputs)
        .def("global_outputs", &Iterator::globalOutputs)
        .def("next", &Iterator::next);
    m.def("refactor_tensor", edge)
        .def("refactor_operator", node)
        .def("refactor_graph", graph);
}
} // namespace

PYBIND11_MODULE(backend, m) { register_refactor(m); }
