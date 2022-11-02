#include "core/graph.h"
#include "operators/extend.h"
#include "operators/gather.h"
#include "operators/reduce_mean.h"
#include "operators/transpose.h"

#include "pfusion/instantiate.h"
#include "pfusion/memory_codegen.h"
#include "pfusion/pointer.h"
#include "pfusion/search_graph.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

void exportCode(const std::string &filename, const std::string &code) {
    // check dir
    if (std::filesystem::exists("../generated_code")) {
        assert(std::filesystem::is_directory("../generated_code"));
    } else {
        auto ok = std::filesystem::create_directory("../generated_code");
        assert(ok);
    }
    std::string dir = "../generated_code/" + filename;
    std::ofstream fout(dir);
    assert(fout.is_open());
    fout << code;
    fout.close();
    system(std::string("clang-format -i " + dir).c_str());
}

void infini::MemoryCodegen::exportGraph(Graph graph, std::string filename) {
    std::string code = generate(graph);
    exportCode(filename, code);
}

void infini::MemoryCodegen::exportBert_LN(const std::string &filename) {
    std::string code = "";
    exportCode(filename, code);
}

void infini::MemoryCodegen::exportBert_SM(const std::string &filename) {
    std::string code = "";
    exportCode(filename, code);
}

void infini::MemoryCodegen::exportBert_GELU(const std::string &filename) {
    std::string code = "";
    exportCode(filename, code);
}

void infini::MemoryCodegen::exportViT_LN(const std::string &filename) {
    std::string code = "";
    exportCode(filename, code);
}

void infini::MemoryCodegen::exportViT_SM(const std::string &filename) {
    std::string code = "";
    exportCode(filename, code);
}

void infini::MemoryCodegen::exportViT_GELU(const std::string &filename) {
    std::string code = "";
    exportCode(filename, code);
}

std::vector<size_t> convertShape(const std::vector<int> &_shape) {
    std::vector<size_t> shape;
    for (int i = int(_shape.size()); i > 0; i--) {
        shape.emplace_back(_shape[i - 1]);
    }
    return shape;
}

size_t convertIndex(const size_t idx, const size_t size) { return size - idx; }

std::vector<size_t> convertPerm(const std::vector<int> &_perm) {
    std::vector<size_t> perm;
    for (int i = int(_perm.size()); i > 0; i--) {
        perm.emplace_back(_perm.size() - _perm[i - 1] - 1);
    }
    return perm;
}

void convertTranspose(std::shared_ptr<memb::SearchGraph> searchGraph,
                      infini::Operator op) {
    searchGraph->addNode(memb::instantiateTranspose(
        memb::TRANSPOSE,
        {memb::Pointer::buildPtrByTensorGuid(op->getInputs()[0]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getOutputs()[0]->getGuid())},
        convertShape(op->getOutputs()[0]->getDims()),
        convertPerm(infini::as<infini::TransposeObj>(op)->getPerm())));
}

void convertUnary(std::shared_ptr<memb::SearchGraph> searchGraph,
                  infini::Operator op, memb::OpType opType) {
    searchGraph->addNode(memb::instantiateUnary(
        opType,
        {memb::Pointer::buildPtrByTensorGuid(op->getInputs()[0]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getOutputs()[0]->getGuid())},
        convertShape(op->getOutputs()[0]->getDims())));
}

void convertBinary(std::shared_ptr<memb::SearchGraph> searchGraph,
                   infini::Operator op, memb::OpType opType) {
    searchGraph->addNode(memb::instantiateBinary(
        opType,
        {memb::Pointer::buildPtrByTensorGuid(op->getInputs()[0]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getInputs()[1]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getOutputs()[0]->getGuid())},
        convertShape(op->getOutputs()[0]->getDims())));
}
void convertGather(std::shared_ptr<memb::SearchGraph> searchGraph,
                   infini::Operator op) {
    searchGraph->addNode(memb::instantiateGather(
        memb::GATHER,
        {memb::Pointer::buildPtrByTensorGuid(op->getInputs()[0]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getInputs()[1]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getOutputs()[0]->getGuid())},
        convertShape(op->getInputs()[0]->getDims()),
        convertShape(op->getInputs()[1]->getDims()),
        convertShape(op->getOutputs()[0]->getDims()),
        convertIndex(infini::as<infini::GatherObj>(op)->getAxis(),
                     op->getInputs()[0]->getDims().size())));
}
void convertReduce(std::shared_ptr<memb::SearchGraph> searchGraph,
                   infini::Operator op, memb::OpType opType) {
    auto reduceMeanOp = infini::as<infini::ReduceMeanObj>(op);
    int axis = -1,
        dimSize = int(reduceMeanOp->getInputs()[0]->getDims().size());
    for (int i = 0; i < dimSize; i++) {
        if (reduceMeanOp->isReduced(i)) {
            if (axis != -1) {
                IT_ASSERT(false);
            } else {
                axis = dimSize - i - 1;
            }
        }
    }
    IT_ASSERT(axis != -1);
    searchGraph->addNode(memb::instantiateReduce(
        opType,
        {memb::Pointer::buildPtrByTensorGuid(op->getInputs()[0]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getOutputs()[0]->getGuid())},
        convertShape(op->getInputs()[0]->getDims()), axis));
}

void convertBroadcast(std::shared_ptr<memb::SearchGraph> searchGraph,
                      infini::Operator op) {
    auto extendOp = infini::as<infini::ExtendObj>(op);
    IT_ASSERT(op->getInputs()[0]->getDims()[extendOp->getDim()] == 1);
    searchGraph->addNode(memb::instantiateBroadcast(
        memb::BROADCAST,
        {memb::Pointer::buildPtrByTensorGuid(op->getInputs()[0]->getGuid()),
         memb::Pointer::buildPtrByTensorGuid(op->getOutputs()[0]->getGuid())},
        convertShape(op->getInputs()[0]->getDims()), extendOp->getDim(),
        extendOp->getNum() + 1));
}

std::shared_ptr<memb::SearchGraph> instantiateGraph(infini::Graph graph) {
    auto searchGraph = std::make_shared<memb::SearchGraph>();
    std::unordered_map<int, int> opMap;
    int id = 0;
    for (auto op : graph->getOperators()) {
        switch (op->getOpType()) {
        case infini::OpType::Transpose:
            convertTranspose(searchGraph, op);
            break;
        case infini::OpType::Relu:
            convertUnary(searchGraph, op, memb::RELU);
            break;
        case infini::OpType::Add:
            convertBinary(searchGraph, op, memb::ADD);
            break;
        case infini::OpType::Sub:
            convertBinary(searchGraph, op, memb::SUB);
            break;
        case infini::OpType::Gather:
            convertGather(searchGraph, op);
            break;
        case infini::OpType::ReduceMean:
            convertReduce(searchGraph, op, memb::REDUCEMEAN);
            break;
        case infini::OpType::Extend:
            convertBroadcast(searchGraph, op);
            break;
        default:
            std::cout << int(op->getOpType()) << std::endl;
            IT_ASSERT(false);
        }
        IT_ASSERT(opMap.find(op->getGuid()) == opMap.end());
        opMap[op->getGuid()] = id;
        id++;
    }
    for (auto op : graph->getOperators()) {
        for (auto nextOp : op->getSuccessors()) {
            assert(opMap.find(op->getGuid()) != opMap.end());
            assert(opMap.find(nextOp->getGuid()) != opMap.end());
            searchGraph->addEdge(opMap[op->getGuid()],
                                 opMap[nextOp->getGuid()]);
        }
    }
    return searchGraph;
}

std::string infini::MemoryCodegen::generate(Graph graph) {
    auto searchGraph = instantiateGraph(graph);
    auto metaGraph = searchGraph->exportFirstMetaGraph();
    std::string code = "";
    std::cout << "[INFO] before opt." << std::endl;
    metaGraph->print();
    metaGraph->optimize();
    std::cout << "[INFO] after opt." << std::endl;
    metaGraph->print();
    code += metaGraph->genHeader();
    code += metaGraph->genKernelFuncs();
    code += metaGraph->genInvokeFuncs();
    return code;
}
