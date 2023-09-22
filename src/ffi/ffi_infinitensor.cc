#include "common/error_handler.h"
#include "communication/operators.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "frontend/graph.h"
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

namespace frontend {
class Compiler;
} // namespace frontend

namespace infini {
class Executor;
} // namespace infini

static std::string getFormat(refactor::common::DataType);

namespace frontend {
using namespace refactor;
using namespace refactor::frontend;
using Name = std::string;

class Compiler {
    Graph _g;

  public:
    explicit Compiler(Graph g) : _g(std::move(g)) {}
    std::unordered_set<Name> fillEdgeInfo() { return _g.fillEdgeInfo(); }
    void setInput(size_t index, int dataType,
                  std::vector<std::variant<std::string, int64_t>> shape) {
        ASSERT(index < _g.internal().topology.globalInputsCount(),
               fmt::format("set input {} failed with wrong index", index));
        auto dataType_ = *common::DataType::parse(dataType);
        Shape shape_(shape.size(), DimExpr(1));
        std::transform(shape.begin(), shape.end(), shape_.begin(),
                       [](auto const &d) -> DimExpr {
                           if (std::holds_alternative<std::string>(d)) {
                               return DimExpr(std::get<std::string>(d));
                           } else {
                               return DimExpr(std::get<int64_t>(d));
                           }
                       });
        _g.internal().edges[index].tensor =
            Tensor::share(dataType_, std::move(shape_), {});
    }
    void substitute(const char *name, int64_t value) {
        ASSERT(_g.substitute(name, value),
               fmt::format("Variable {} not exist", name));
    }
    auto const &graph() const { return _g.internal(); }

    infini::Executor compile(infini::Runtime);

    py::array getTensor(size_t i) {
        auto const &tensor = *_g.internal().edges.at(i).tensor;

        std::vector<int64_t> shape(tensor.shape.size());
        std::transform(tensor.shape.begin(), tensor.shape.end(), shape.begin(),
                       [](auto const &d) { return d.value(); });

        auto ans = py::array(py::dtype(getFormat(tensor.dataType)),
                             std::move(shape), nullptr);
        if (tensor.data) {
            std::memcpy(ans.mutable_data(), tensor.data->ptr, ans.nbytes());
        }
        return ans;
    }
};
} // namespace frontend

namespace infini {

class Executor {
    infini::Graph _g;
    std::vector<infini::Tensor> _inputs;
    std::vector<std::pair<refactor::frontend::Edge, infini::Tensor>> _outputs;

  public:
    Executor(infini::Runtime rt, refactor::frontend::Graph const &frontend)
        : _g(infini::make_ref<infini::GraphObj>(std::move(rt))) {
        using namespace infini;

        auto frontendOutputs = frontend.internal().topology.globalOutputs();
        auto outputsCount = frontendOutputs.size();

        auto [hwInputs, hwOutputs] =
            _g->transformFromGraphTopo(frontend, _g->getRuntime());

        _inputs = std::move(hwInputs);

        ASSERT(outputsCount == hwOutputs.size(), "Output size mismatch");
        _outputs.reserve(outputsCount);

        for (size_t i = 0; i < hwOutputs.size(); ++i) {
            _outputs.emplace_back(frontend.internal().edges[frontendOutputs[i]],
                                  std::move(hwOutputs[i]));
        }
    }

    void run() { _g->getRuntime()->run(_g); }

    void setInput(size_t i, py::array data) {
        _inputs.at(i)->copyin(data.data(), data.nbytes());
    }
    py::array getOutput(size_t i) {
        auto const &[edge, tensor] = _outputs.at(i);

        std::vector<int64_t> shape(edge.tensor->shape.size());
        std::transform(edge.tensor->shape.begin(), edge.tensor->shape.end(),
                       shape.begin(), [](auto const &d) { return d.value(); });

        auto ans = py::array(py::dtype(getFormat(edge.tensor->dataType)),
                             std::move(shape), nullptr);
        if (tensor) {
            tensor->copyout(ans.mutable_data(), ans.nbytes());
        } else if (edge.tensor->data) {
            std::memcpy(ans.mutable_data(), edge.tensor->data->ptr,
                        ans.nbytes());
        }
        return ans;
    }
};

} // namespace infini

namespace frontend {

infini::Executor Compiler::compile(infini::Runtime rt) {
    _g.collectVariables();
    ASSERT(_g.fillEdgeInfo().empty(), "Unknown variables");
    return infini::Executor(std::move(rt), _g);
}

using TExport = std::tuple<Name, int, std::vector<std::variant<Name, int>>,
                           std::optional<py::array>>;
using OExport =
    std::tuple<Name, Name, std::unordered_map<Name, decltype(Attribute::value)>,
               std::vector<Name>, std::vector<Name>>;

class NodeExport {
    std::shared_ptr<Compiler> _internal;
    graph_topo::GraphTopo::Iterator _it;

  public:
    explicit NodeExport(std::shared_ptr<Compiler> internal)
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
        auto outputs = _it.globalOutputs();
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
        auto const opType_ = node.op.opType.name();
        auto opType =
            opType_.substr(0, 6) == "onnx::" ? opType_.substr(6) : opType_;
        auto const &attributes = node.op.attributes;
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
    std::shared_ptr<Compiler> _internal;
    size_t _i;

  public:
    explicit EdgeExport(std::shared_ptr<Compiler> internal)
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
            if (edge.tensor->data) {
                return std::make_tuple(
                    edge.name, static_cast<int>(edge.tensor->dataType.internal),
                    std::move(shape_), _internal->getTensor(_i - 1));
            } else {
                return std::make_tuple(
                    edge.name, static_cast<int>(edge.tensor->dataType.internal),
                    std::move(shape_), std::nullopt);
            }
        }
        return std::nullopt;
    }
};

std::shared_ptr<Tensor> edge(int dataType, std::vector<DimExpr> shape,
                             std::optional<py::array> data) {
    auto ans = Tensor::share(*common::DataType::parse(dataType),
                             Shape(shape.begin(), shape.end()), {});
    if (data) {
        auto const bytesSize = ans->bytesSize();
        ASSERT(bytesSize == static_cast<size_t>(data->nbytes()),
               "Data size mismatch");
        ans->data = std::make_shared<common::Blob>(bytesSize);
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

std::shared_ptr<Compiler>
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
        auto node = Node{std::move(*operator_), name};
        builder.nodes.insert({std::move(name), std::move(node)});
    }
    for (auto &[name, tensor] : edges) {
        auto edge = Edge{std::move(tensor), name};
        auto it = builder.edges.find(name);
        ASSERT(it != builder.edges.end(),
               fmt::format("edge {} not connected", name));
        it->second.tensor = std::move(edge.tensor);
    }
    builder.globalInputs = std::move(inputs);
    builder.globalOutputs = std::move(outputs);
    return std::make_shared<Compiler>(Graph(builder.build()));
}

void registerRefactor(py::module &m) {
    using policy = py::return_value_policy;

    onnx::register_();
    communication::register_();

    py::class_<DimExpr>(m, "DimExpr")
        .def(py::init<int64_t>())
        .def(py::init<std::string>());
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator");
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor");
    py::class_<Compiler, std::shared_ptr<Compiler>>(m, "Graph")
        .def("fill_edge_info", &Compiler::fillEdgeInfo, policy::move)
        .def("substitute", &Compiler::substitute, policy::automatic)
        .def("set_input", &Compiler::setInput, policy::automatic)
        .def("compile", &Compiler::compile, policy::move)
        .def("get_tensor", &Compiler::getTensor, policy::move);
    py::class_<NodeExport>(m, "NodeExport")
        .def(py::init<std::shared_ptr<Compiler>>())
        .def("global_inputs", &NodeExport::globalInputs, policy::move)
        .def("global_outputs", &NodeExport::globalOutputs, policy::move)
        .def("next", &NodeExport::next, policy::move);
    py::class_<EdgeExport>(m, "EdgeExport")
        .def(py::init<std::shared_ptr<Compiler>>())
        .def("next", &EdgeExport::next, policy::move);
    m.def("refactor_tensor", edge, policy::move)
        .def("refactor_operator", node, policy::move)
        .def("refactor_graph", graph, policy::move);
}
} // namespace frontend

namespace infini {
using policy = py::return_value_policy;

infini::Runtime cpuRuntime() {
    return std::dynamic_pointer_cast<RuntimeObj>(
        NativeCpuRuntimeObj::getInstance());
}

infini::Runtime cudaRuntime(int deviceId) {
#ifdef USE_CUDA
    return std::dynamic_pointer_cast<RuntimeObj>(
        make_ref<CudaRuntimeObj>(deviceId));
#endif
    return nullptr;
}

void registerPy(py::module &m) {
    py::class_<Executor>(m, "Executor")
        .def("run", &Executor::run, policy::automatic)
        .def("set_input", &Executor::setInput, policy::move)
        .def("get_output", &Executor::getOutput, policy::move);
    py::class_<RuntimeObj, Runtime>(m, "Runtime")
        .def("init_comm", &RuntimeObj::initComm, policy::automatic);
    m.def("cpu_runtime", cpuRuntime, policy::move)
        .def("cuda_runtime", cudaRuntime, policy::move);
}
} // namespace infini

// A helper function that converts DataType to python format string
static std::string getFormat(refactor::common::DataType type) {
    using namespace refactor::common;

#define CASE(T)                                                                \
    case DataType::T:                                                          \
        return py::format_descriptor<primitive_t<DataType::T>::type>::format();

    switch (type.internal) {
        CASE(F32);
        CASE(F64);
        CASE(I32);
        CASE(I64);
        CASE(I8);
        CASE(I16);
        CASE(U8);
        CASE(U16);
        CASE(U32);
        CASE(U64);
    case DataType::FP16:
    case DataType::BF16:
        // Python uses "e" for half precision float type code.
        // Check the following link for more information.
        // https://docs.python.org/3/library/struct.html#format-characters
        return "e";
    default:
        RUNTIME_ERROR("unsupported data type.");
    }
}

PYBIND11_MODULE(backend, m) {
    frontend::registerRefactor(m);
    infini::registerPy(m);
}
