#include "common/error_handler.h"
#include "communication/operators.h"
#include "computation/graph.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "onnx/operators.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#include "cuda/operator_timer.h"
#endif
#ifdef USE_BANG
#include "bang/bang_runtime.h"
#endif
#ifdef USE_INTELCPU
#include "intelcpu/mkl_runtime.h"
#include "intelcpu/operator_timer.h"
#endif

namespace py = pybind11;

namespace {
using namespace refactor;
using namespace computation;
using Name = std::string;

class Handler {
    Graph _g;
    std::vector<infini::Tensor> _outputs;

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

    void runCuda() {
        using namespace infini;
#ifdef USE_CUDA
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        auto graph = make_ref<GraphObj>(cudaRuntime);
        _outputs = graph->transformFromGraphTopo(_g, cudaRuntime);
        graph->getRuntime()->run(graph);
#endif
    }
    template <class T> std::vector<T> copyout(size_t i) {
        return _outputs[i]->copyout<T>();
    }
};

using TExport = std::tuple<Name, int, std::vector<std::variant<Name, int>>>;
using OExport =
    std::tuple<Name, Name, std::unordered_map<Name, decltype(Attribute::value)>,
               std::vector<Name>, std::vector<Name>>;

class NodeExport {
    std::shared_ptr<Handler> _internal;
    graph_topo::GraphTopo::Iterator _it;

  public:
    explicit NodeExport(std::shared_ptr<Handler> internal)
        : _internal(std::move(internal)),
          _it(_internal->graph().topology.begin()) {}

    std::vector<Name> globalInputs() const {
        auto inputs = _it.globalInputs();
        std::vector<Name> ans(inputs.size());
        std::transform(
            inputs.begin(), inputs.end(), ans.begin(),
            [this](auto const &i) { return _internal->graph().edges[i].name; });
        return ans;
    }

    std::vector<Name> globalOutputs() const {
        auto outputs = _it.globalInputs();
        std::vector<Name> ans(outputs.size());
        std::transform(
            outputs.begin(), outputs.end(), ans.begin(),
            [this](auto const &i) { return _internal->graph().edges[i].name; });
        return ans;
    }

    std::optional<OExport> next() {
        if (_it == _internal->graph().topology.end()) {
            return std::nullopt;
        }
        auto [nodeIdx, inputs_, outputs_] = *_it++;
        auto const &node = _internal->graph().nodes[nodeIdx];
        auto const &name = node.name;
        auto const opType_ = node.op->opType.name();
        auto opType =
            opType_.substr(0, 6) == "onnx::" ? opType_.substr(6) : opType_;
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
        return OExport{name, opType, attributes_, inputs, outputs};
    }
};

class EdgeExport {
    std::shared_ptr<Handler> _internal;
    size_t _i;

  public:
    explicit EdgeExport(std::shared_ptr<Handler> internal)
        : _internal(std::move(internal)), _i(0) {}

    std::optional<TExport> next() {
        while (_i != _internal->graph().edges.size()) {
            auto const &edge = _internal->graph().edges[_i++];
            if (!edge.tensor) {
                continue;
            }
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
            return TExport{edge.name, static_cast<int>(edge.tensor->dataType),
                           std::move(shape_)};
        }
        return std::nullopt;
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
    communication::register_();

    py::class_<DimExpr>(m, "DimExpr")
        .def(py::init<int64_t>())
        .def(py::init<std::string>());
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator");
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor");
    py::class_<Handler, std::shared_ptr<Handler>>(m, "Graph")
        .def("fill_edge_info", &Handler::fillEdgeInfo)
        .def("substitute", &Handler::substitute)
        .def("run_cuda", &Handler::runCuda)
        .def("copy_out_float", &Handler::copyout<float>)
        .def("set_input", &Handler::setInput);
    py::class_<NodeExport>(m, "NodeExport")
        .def(py::init<std::shared_ptr<Handler>>())
        .def("global_inputs", &NodeExport::globalInputs)
        .def("global_outputs", &NodeExport::globalOutputs)
        .def("next", &NodeExport::next);
    py::class_<EdgeExport>(m, "EdgeExport")
        .def(py::init<std::shared_ptr<Handler>>())
        .def("next", &EdgeExport::next);
    m.def("refactor_tensor", edge)
        .def("refactor_operator", node)
        .def("refactor_graph", graph);
}
} // namespace

PYBIND11_MODULE(backend, m) { register_refactor(m); }
