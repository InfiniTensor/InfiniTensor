#include "core/graph.h"
#include "operators/transpose.h"

#include "pfusion/instantiate.h"
#include "pfusion/memory_codegen.h"
#include "pfusion/pointer.h"
#include "pfusion/search_graph.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

void infini::MemoryCodegen::exportCode(Graph graph, std::string filename) {
    // check dir
    if (std::filesystem::exists("../generated_code")) {
        assert(std::filesystem::is_directory("../generated_code"));
    } else {
        auto ok = std::filesystem::create_directory("../generated_code");
        assert(ok);
    }

    // generate code
    std::string dir = "../generated_code/" + filename;
    std::ofstream fout(dir);
    assert(fout.is_open());
    fout << generate(graph);
    fout.close();
    system(std::string("clang-format -i " + dir).c_str());
}

std::vector<size_t> convertShape(const std::vector<int> &_shape) {
    std::vector<size_t> shape;
    for (int i = int(_shape.size()); i > 0; i--) {
        shape.emplace_back(_shape[i - 1]);
    }
    return shape;
}

std::vector<size_t> convertPerm(const std::vector<int> &_perm) {
    std::vector<size_t> perm;
    for (int i = int(_perm.size()); i > 0; i--) {
        perm.emplace_back(_perm.size() - _perm[i - 1] - 1);
    }
    return perm;
}

std::shared_ptr<memb::SearchGraph> instantiateGraph(infini::Graph graph) {
    auto metaGraph = std::make_shared<memb::SearchGraph>();
    std::unordered_map<int, int> opMap;
    int id = 0;
    for (auto op : graph->getOperators()) {
        switch (op->getOpType()) {
        case infini::OpType::Transpose:
            metaGraph->addNode(memb::instantiateTranspose(
                memb::TRANSPOSE,
                {memb::Pointer::buildPtrByTensorGuid(
                     op->getInputs()[0]->getGuid()),
                 memb::Pointer::buildPtrByTensorGuid(
                     op->getOutputs()[0]->getGuid())},
                convertShape(op->getOutputs()[0]->getDims()),
                convertPerm(infini::as<infini::TransposeObj>(op)->getPerm())));
            break;
        case infini::OpType::Relu:
            metaGraph->addNode(memb::instantiateUnary(
                memb::RELU,
                {memb::Pointer::buildPtrByTensorGuid(
                     op->getInputs()[0]->getGuid()),
                 memb::Pointer::buildPtrByTensorGuid(
                     op->getOutputs()[0]->getGuid())},
                convertShape(op->getOutputs()[0]->getDims())));
            break;
        case infini::OpType::Add:
            metaGraph->addNode(memb::instantiateBinary(
                memb::ADD,
                {memb::Pointer::buildPtrByTensorGuid(
                     op->getInputs()[0]->getGuid()),
                 memb::Pointer::buildPtrByTensorGuid(
                     op->getInputs()[1]->getGuid()),
                 memb::Pointer::buildPtrByTensorGuid(
                     op->getOutputs()[0]->getGuid())},
                convertShape(op->getOutputs()[0]->getDims())));
            break;
        case infini::OpType::Sub:
            metaGraph->addNode(memb::instantiateBinary(
                memb::SUB,
                {memb::Pointer::buildPtrByTensorGuid(
                     op->getInputs()[0]->getGuid()),
                 memb::Pointer::buildPtrByTensorGuid(
                     op->getInputs()[1]->getGuid()),
                 memb::Pointer::buildPtrByTensorGuid(
                     op->getOutputs()[0]->getGuid())},
                convertShape(op->getOutputs()[0]->getDims())));
            break;
        default:
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
            metaGraph->addEdge(opMap[op->getGuid()], opMap[nextOp->getGuid()]);
        }
    }
    return metaGraph;
}

std::string infini::MemoryCodegen::generate(Graph graph) {
    auto searchGraph = instantiateGraph(graph);
    auto metaGraph = searchGraph->exportFirstMetaGraph();
    std::string code = "";
    code += metaGraph->genHeader();
    code += metaGraph->genKernelFuncs();
    code += metaGraph->genInvokeFuncs();
    return code;
}
