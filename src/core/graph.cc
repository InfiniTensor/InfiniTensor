#include "core/graph.h"
#include "operators/element_wise.h"
#include "operators/layer_norm.h"
#include "operators/matmul.h"
#include "operators/reduce.h"
#include "operators/reshape.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace std {
template <> struct hash<infini::OpType> {
    std::size_t operator()(const infini::OpType &opType) const {
        return std::hash<infini::OpType::underlying_t>()(opType.underlying());
    }
};
} // namespace std

namespace infini {

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), allocator(runtime), sorted(false) {
    map<UidBaseType, Tensor> tensorPool;
    // Clone tensors
    for (const auto &op : ops_in) {
        for (const auto &t : op->getInputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
        for (const auto &t : op->getOutputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
    }
    // Clone operators and add connections
    for (const auto &op : ops_in) {
        TensorVec inputs, outputs;
        for (const auto &t : op->getInputs()) {
            if (t) {
                inputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }

        for (const auto &t : op->getOutputs()) {
            if (t) {
                outputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }
        addOperatorAndConnect(op->clone(inputs, outputs));
    }
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() &&
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
    this->ops = std::move(sorted);
    return this->sorted = true;
}

void GraphObj::optimize() {
#ifndef USE_REFACOTRGRAPH
    for (auto &op : ops) {
        switch (op->getOpType().underlying()) {
        default:
            break;
        }
    }
#else
    fmt::println("Convert to RefactorGraph.");
    refactor::computation::Graph refactorGraph = convertToRefactorGraph();
    auto const &g_ = refactorGraph.internal().contiguous();
    fmt::println("Before optimization.");
    fmt::println("{}", g_.topology.toString());
    fmt::println("Nodes info :");
    for (size_t i = 0; i < g_.nodes.size(); ++i) {
        fmt::println("{}. \"{}\"", i, g_.nodes[i].name);
    }
    fmt::println("\n Edges info :");
    for (size_t i = 0; i < g_.edges.size(); ++i) {
        fmt::println("{}. \"{}\" Shape is {}, Layout is {}", i,
                     g_.edges[i].name,
                     refactor::vec2str(g_.edges[i].tensor->shape),
                     g_.edges[i].tensor->layout.name());
    }
    refactorGraph.optimize();
    fmt::println("After optimization.");
    auto const &g_x = refactorGraph.internal().contiguous();
    fmt::println("{}", g_x.topology.toString());
    fmt::println("Nodes info :");
    for (size_t i = 0; i < g_x.nodes.size(); ++i) {
        fmt::println("{}. \"{}\"", i, g_x.nodes[i].name);
    }
    fmt::println("\n Edges info :");
    for (size_t i = 0; i < g_x.edges.size(); ++i) {
        fmt::println("{}. \"{}\" Shape is {}, Layout is {}", i,
                     g_x.edges[i].name,
                     refactor::vec2str(g_x.edges[i].tensor->shape),
                     g_x.edges[i].tensor->layout.name());
    }

    // Convert back to InfiniGraph
    fmt::println("Convert back to InfiniGraph.");
    this->convertFromRefactorGraph(refactorGraph);
    fmt::println("Optimization done.");
    fmt::println("{}", this->toString());
#endif
}

#ifdef USE_REFACOTRGRAPH
using NodeFactory =
    std::function<refactor::computation::Node(const Operator &)>;

using OpFactory = std::function<void(
    const refactor::computation::Node &, refactor::slice_t<refactor::count_t>,
    refactor::range_t<refactor::count_t>, std::unordered_map<size_t, Tensor> &,
    GraphObj &)>;

#define DEFINE_BINARY_OP(op_type, op_name)                                     \
    {                                                                          \
        op_type, [](const Operator &op) {                                      \
            return refactor::computation::Node{                                \
                std::make_unique<refactor::computation::SimpleBinary>(         \
                    refactor::computation::SimpleBinaryType::op_name),         \
                #op_name};                                                     \
        }                                                                      \
    }

#define DEFINE_UNARY_OP(op_type, op_name)                                      \
    {                                                                          \
        op_type, [](const Operator &op) {                                      \
            return refactor::computation::Node{                                \
                std::make_unique<refactor::computation::SimpleUnary>(          \
                    refactor::computation::SimpleUnaryType::op_name),          \
                #op_name};                                                     \
        }                                                                      \
    }

#define DEFINE_REDUCE_OP(op_type, op_class, reduce_type)                       \
    {                                                                          \
        op_type, [](const Operator &_op) {                                     \
            auto op = as<op_class>(_op);                                       \
            std::set<int> axesSet = op->getAxes();                             \
            absl::InlinedVector<uint32_t, 4> axes(axesSet.begin(),             \
                                                  axesSet.end());              \
            return refactor::computation::Node{                                \
                std::make_unique<refactor::computation::Reduce>(               \
                    refactor::computation::ReduceType::reduce_type, axes, 3,   \
                    op->getKeepDims()),                                        \
                "Reduce" #reduce_type};                                        \
        }                                                                      \
    }

#define DEFINE_SIMPLE_BINARY_OP_BACK(OpClass, ...)                             \
    {                                                                          \
        refactor::computation::SimpleBinary::typeId(__VA_ARGS__),              \
            [](const refactor::computation::Node &node,                        \
               refactor::slice_t<refactor::count_t> inputs,                    \
               refactor::range_t<refactor::count_t> outputs,                   \
               std::unordered_map<size_t, Tensor> &tensorMap, GraphObj &g) {   \
                g.addOpWithOutputs<OpClass>(tensorMap[inputs[0]],              \
                                            tensorMap[inputs[1]],              \
                                            tensorMap[outputs[0]]);            \
            }                                                                  \
    }

std::unordered_map<OpType, NodeFactory> nodeFactoryMap = {
    DEFINE_BINARY_OP(OpType::Add, Add),
    DEFINE_BINARY_OP(OpType::Sub, Sub),
    DEFINE_BINARY_OP(OpType::Mul, Mul),
    DEFINE_BINARY_OP(OpType::Div, Div),
    DEFINE_BINARY_OP(OpType::Pow, Pow),
    DEFINE_BINARY_OP(OpType::And, And),
    DEFINE_BINARY_OP(OpType::Or, Or),
    DEFINE_BINARY_OP(OpType::Xor, Xor),
    DEFINE_BINARY_OP(OpType::Mod, Mod),
    DEFINE_BINARY_OP(OpType::FloorMod, Fmod),
    DEFINE_UNARY_OP(OpType::Abs, Abs),
    DEFINE_UNARY_OP(OpType::Acos, Acos),
    DEFINE_UNARY_OP(OpType::Acosh, Acosh),
    DEFINE_UNARY_OP(OpType::Asin, Asin),
    DEFINE_UNARY_OP(OpType::Asinh, Asinh),
    DEFINE_UNARY_OP(OpType::Atan, Atan),
    DEFINE_UNARY_OP(OpType::Atanh, Atanh),
    DEFINE_UNARY_OP(OpType::Cos, Cos),
    DEFINE_UNARY_OP(OpType::Cosh, Cosh),
    DEFINE_UNARY_OP(OpType::Sin, Sin),
    DEFINE_UNARY_OP(OpType::Sinh, Sinh),
    DEFINE_UNARY_OP(OpType::Tan, Tan),
    DEFINE_UNARY_OP(OpType::Tanh, Tanh),
    DEFINE_UNARY_OP(OpType::Relu, Relu),
    DEFINE_UNARY_OP(OpType::Sqrt, Sqrt),
    DEFINE_UNARY_OP(OpType::Sigmoid, Sigmoid),
    DEFINE_UNARY_OP(OpType::Erf, Erf),
    DEFINE_UNARY_OP(OpType::Neg, Neg),
    DEFINE_UNARY_OP(OpType::Not, Not),
    DEFINE_UNARY_OP(OpType::HardSwish, HardSwish),
    DEFINE_UNARY_OP(OpType::Exp, Exp),
    DEFINE_REDUCE_OP(OpType::ReduceMean, ReduceMeanObj, Mean),
    DEFINE_REDUCE_OP(OpType::ReduceSum, ReduceSumObj, Sum),
    {OpType::Conv,
     [](const Operator &op) {
         return refactor::computation::Node{
             std::make_unique<refactor::computation::Conv>(
                 refactor::kernel::PoolAttributes(2, nullptr, nullptr,
                                                  nullptr)),
             "conv"};
     }},
    {OpType::MatMul,
     [](const Operator &_op) {
         auto op = as<MatmulObj>(_op);
         return refactor::computation::Node{
             std::make_unique<refactor::computation::MatMul>(
                 1.0f, 1.0f, op->getTransA(), op->getTransB()),
             "matmul"};
     }},
    {OpType::Transpose, [](const Operator &op) {
         auto transpose = as<TransposeObj>(op);
         auto permute = transpose->getPermute();
         return refactor::computation::Node{
             std::make_unique<refactor::computation::Transpose>(
                 absl::InlinedVector<refactor::dim_t, 4>(permute.begin(),
                                                         permute.end())),
             "transpose"};
     }}};

std::unordered_map<size_t, OpFactory> opHandlers = {
    {refactor::computation::MatMul::typeId(),
     [](const refactor::computation::Node &node,
        refactor::slice_t<refactor::count_t> inputs,
        refactor::range_t<refactor::count_t> outputs,
        std::unordered_map<size_t, Tensor> &tensorMap, GraphObj &g) {
         auto op =
             dynamic_cast<refactor::computation::MatMul const *>(node.op.get());
         g.addOpWithOutputs<MatmulObj>(
             tensorMap[inputs[0]], tensorMap[inputs[1]], tensorMap[outputs[0]],
             op->transA, op->transB);
     }},
    {refactor::computation::LayerNormalization::typeId(),
     [](const refactor::computation::Node &node,
        refactor::slice_t<refactor::count_t> inputs,
        refactor::range_t<refactor::count_t> outputs,
        std::unordered_map<size_t, Tensor> &tensorMap, GraphObj &g) {
         auto op =
             dynamic_cast<refactor::computation::LayerNormalization const *>(
                 node.op.get());
         g.addOpWithOutputs<LayerNormObj>(
             tensorMap[inputs[0]], tensorMap[inputs[1]], tensorMap[outputs[0]],
             tensorMap[inputs[2]], op->epsilon, op->axis);
     }},
    DEFINE_SIMPLE_BINARY_OP_BACK(AddObj,
                                 refactor::computation::SimpleBinaryType::Add),
};

refactor::Arc<refactor::kernel::Tensor> convertToRefactorTensor(Tensor tensor) {
    Shape shape = tensor->getDims();
    int dtype = tensor->getDTypeIndex();
    auto layout = refactor::kernel::LayoutType::Others;
    absl::InlinedVector<refactor::dim_t, 4> ref_shape(shape.begin(),
                                                      shape.end());
    return refactor::kernel::Tensor::share(*refactor::DataType::parse(dtype),
                                           ref_shape, layout);
}

void GraphObj::convertFromRefactorGraph(
    const refactor::computation::Graph &refactorGraph) {
    auto const &g_ = refactorGraph.internal().contiguous();
    std::unordered_map<size_t, Tensor> tensorMap;
    size_t tensorId = 0;
    for (auto const &edge : g_.edges) {
        auto shape = edge.tensor->shape;
        tensorMap[tensorId++] =
            this->addTensor(std::vector<int>(shape.begin(), shape.end()),
                            DataType(static_cast<int>(edge.tensor->dataType)));
    }
    for (auto const &item : g_.topology) {
        refactor::computation::Node const &node = g_.nodes[item.idx];
        auto const &op = node.op;
        auto const &inputs = item.inputs;
        auto const &outputs = item.outputs;
        opHandlers[op->opTypeId()](node, inputs, outputs, tensorMap, *this);
    }
}

refactor::computation::Graph GraphObj::convertToRefactorGraph() {
    auto nodes = std::unordered_map<size_t, refactor::computation::Node>{};
    auto edges = std::unordered_map<size_t, refactor::computation::Edge>{};
    this->topo_sort();
    TensorVec globalInputs = this->getInputs();
    TensorVec globalOutputs = this->getOutputs();
    TensorVec tensors = this->getTensors();
    std::vector<size_t> globalInputIds;
    std::vector<size_t> globalOutputIds;
    for (auto &input : globalInputs) {
        globalInputIds.push_back(input->getFuid());
    }
    for (auto &output : globalOutputs) {
        globalOutputIds.push_back(output->getFuid());
    }
    for (auto &tensor : tensors) {
        auto tensorId = tensor->getFuid();
        auto tensorData = convertToRefactorTensor(tensor);
        auto tensorName = "tensor_" + std::to_string(tensorId);
        edges[tensorId] = {tensorData, tensorName};
    }
    OpVec ops = this->getOperators();
    std::unordered_map<size_t, refactor::graph_topo::BuilderNode<size_t>>
        topology = {};

    size_t ids = 0;
    for (auto &op : ops) {
        auto inputs = op->getInputs();
        auto outputs = op->getOutputs();
        std::vector<size_t> inputIds;
        std::vector<size_t> outputIds;
        for (auto &input : inputs) {
            inputIds.push_back(input->getFuid());
        }
        for (auto &output : outputs) {
            outputIds.push_back(output->getFuid());
        }
        topology[ids] = {inputIds, outputIds};
        auto _optype = op->getOpType();
        if (nodeFactoryMap.find(_optype) != nodeFactoryMap.end()) {
            nodes[ids] = nodeFactoryMap[_optype](op);
        } else {
            throw std::runtime_error("Unsupported OpType");
        }
        ids++;

        // remove the operator from the graph
        this->removeOperator(op);
    }
    for (auto &tensor : tensors) {
        this->removeTensor(tensor);
    }
    auto graphTopo =
        refactor::graph_topo::Builder<size_t, refactor::computation::Node,
                                      size_t, refactor::computation::Edge>{
            topology,         globalInputIds,   globalOutputIds,
            std::move(nodes), std::move(edges),
        }
            .build();
    return refactor::computation::Graph(std::move(graphTopo));
}
#endif

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::dataMalloc(bool useNaiveAllocator, size_t memPoolSize) {
    // topological sorting first

    IT_ASSERT(topo_sort() == true);
    if (useNaiveAllocator) {
        // can not set memory pool when use naive allocator
        IT_ASSERT(memPoolSize == 0);
        // used for debugging memory out-of-bounds access, tensors will not
        // be released correctly note: behavior may not match running in
        // non-naive mode, and it may not reproduce the bug
        for (auto &tensor : tensors) {
            if (!tensor->isWeight() ||
                (tensor->isWeight() && !weightAllocated)) {
                tensor->dataMalloc();
            }
        }
        return;
    }
    if (memPoolSize > 0) {
        allocator.setMemPool(memPoolSize);
    }
    // count the number of times all tensors are used
    std::unordered_map<TensorObj *, size_t> tensorToRefCount;
    // record the memory address offsets of all tensors to be allocated
    std::unordered_map<TensorObj *, size_t> tensorToOffset;

    // reinit allocator
    allocator.init();

    // record all weight tensors, including weight tensors and kvcache
    // tensors
    std::unordered_set<TensorObj *> weightTensors;
    for (auto &tensor : tensors) {
        if (tensor->isWeight()) {
            // allocate memory for all weight tensors first, and this memory
            // will not be freed until the graph is destroyed
            weightTensors.insert(tensor.get());
            if (!this->weightAllocated) {
                tensorToOffset[tensor.get()] =
                    allocator.allocWeight(tensor->getBytes());
            }
        } else if (tensor->isInput() || tensor->isOutput()) {
            // allocate memory for all input and output tensors, and this
            // memory will not be reused later
            tensorToOffset[tensor.get()] = allocator.alloc(tensor->getBytes());
        } else {
            tensorToRefCount[tensor.get()] = tensor->getTargets().size();
            // allocate memory for all user-created tensors
            if (tensor.get()->getSource() == nullptr) {
                tensorToOffset[tensor.get()] =
                    allocator.alloc(tensor->getBytes());
            }
        }
    }
    // if memory has not yet been allocated for weight tensors,
    // allocate memory now and do not allocate again in the future.
    if (!this->weightAllocated) {
        this->weightAllocated = true;
        // only allocate once for weight tensors
        for (auto &tensor : weightTensors) {
            IT_ASSERT(tensorToOffset.find(tensor) != tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getWeightPtr()) +
                    tensorToOffset[tensor]));
        }
    }
    // traverse in topological order and simulate memory allocation
    for (auto &op : ops) {
        // memory should be allocated for the op's output first
        auto outputs = op->getOutputs();
        for (auto &tensor : outputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    tensorToOffset[tensor.get()] =
                        allocator.alloc(tensor->getBytes());
                }
            }
        }
        auto inputs = op->getInputs();
        for (auto &tensor : inputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    auto tensorIter = tensorToRefCount.find(tensor.get());
                    IT_ASSERT(tensorIter != tensorToRefCount.end());
                    IT_ASSERT(tensorToRefCount[tensor.get()] > 0);
                    tensorToRefCount[tensor.get()] -= 1;
                    if (tensorToRefCount[tensor.get()] == 0) {
                        // indicate that this tensor will no longer be used
                        // and perform memory free
                        tensorToRefCount.erase(tensor.get());
                        allocator.free(tensorToOffset[tensor.get()],
                                       tensor->getBytes());
                    }
                }
            }
        }
    }

    // perform actual memory allocation for non-weight tensors
    for (auto &tensor : tensors) {
        if (!tensor->isWeight()) {
            IT_ASSERT(tensorToOffset.find(tensor.get()) !=
                      tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime, static_cast<uint8_t *>(allocator.getPtr()) +
                                     tensorToOffset[tensor.get()]));
        }
    }
}

Tensor GraphObj::cloneKV(Tensor &tensor) {
    auto obj = tensor->clone();
    if (allocator.getMemPoolStatus()) {
        if (tensor->hasData()) {
            obj->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getHeapPtr()) +
                    allocator.heapAlloc(tensor->getBytes())));
            obj->copyData(tensor);
        }
    } else {
        if (tensor->hasData()) {
            obj->dataMalloc();
            obj->copyData(tensor);
        }
    }
    return obj;
}

void GraphObj::freeHeap() { this->allocator.freeHeap(); }

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

OpVec GraphObj::getComputeOps() const {
    OpVec opList;
    for (auto op : ops)
        if (op->getOpType().isMatMulOrConv())
            opList.emplace_back(op);
    return opList;
}

void GraphObj::deleteConnection(Tensor tensor, Operator op) {
    // if op is target
    IT_ASSERT(std::find(tensor->getTargets().begin(),
                        tensor->getTargets().end(),
                        op) != tensor->getTargets().end());
    tensor->removeTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->removeSuccessors(op);
        op->removePredecessors(tensor->getSource());
    }
}

// add op as a target
void GraphObj::addConnection(Tensor tensor, Operator op) {
    tensor->addTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->addSuccessors(op);
        op->addPredecessors(tensor->getSource());
    }
}

void GraphObj::replaceConnection(Tensor oldTensor, Tensor newTensor,
                                 Operator op) {
    // op is a target of old tensor
    IT_ASSERT(std::find(oldTensor->getTargets().begin(),
                        oldTensor->getTargets().end(),
                        op) != oldTensor->getTargets().end());
    addConnection(newTensor, op);
    deleteConnection(oldTensor, op);
    op->replaceInput(oldTensor, newTensor);
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                    nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini
