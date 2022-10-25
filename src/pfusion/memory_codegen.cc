#include "pfusion/memory_codegen.h"
#include "core/graph.h"
#include "operators/transpose.h"
#include "pfusion/common.h"
#include "pfusion/instantiate.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

void infini::MemoryCodegen::export_code(Graph graph, std::string filename) {
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

std::vector<int> convertShape(const std::vector<int> &_shape) {
    std::vector<int> shape;
    for (int i = int(_shape.size()); i > 0; i--) {
        shape.emplace_back(_shape[i - 1]);
    }
    return shape;
}

std::vector<int> convertPerm(const std::vector<int> &_perm) {
    std::vector<int> perm;
    for (int i = int(_perm.size()); i > 0; i--) {
        perm.emplace_back(_perm.size() - _perm[i - 1] - 1);
    }
    return perm;
}

memb::MetaGraph instantiateGraph(infini::Graph graph) {
    memb::MetaGraph metaGraph;
    std::unordered_map<int, int> opMap;
    int id = 0;
    for (auto op : graph->getOperators()) {
        switch (op->getOpType()) {
        case infini::OpType::Transpose:
            metaGraph.addNode(memb::instantiateTranspose(
                convertShape(op->getOutputs()[0]->getDims()),
                convertPerm(infini::as<infini::TransposeObj>(op)->getPerm())));
            break;
        case infini::OpType::Relu:
            metaGraph.addNode(memb::instantiateUnary(
                convertShape(op->getInputs()[0]->getDims()),
                memb::OpType::RELU));
            break;
        case infini::OpType::Add:
            metaGraph.addNode(memb::instantiateBinary(
                convertShape(op->getInputs()[0]->getDims()),
                memb::OpType::ADD));
            break;
        case infini::OpType::Sub:
            metaGraph.addNode(memb::instantiateBinary(
                convertShape(op->getInputs()[0]->getDims()),
                memb::OpType::SUB));
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
            metaGraph.addEdge(opMap[op->getGuid()], opMap[nextOp->getGuid()]);
        }
    }
    return metaGraph;
}

std::string infini::MemoryCodegen::generate(Graph graph) {
    memb::MetaGraph metaGraph = instantiateGraph(graph);
    metaGraph.print();
    std::string code = "";
    code += metaGraph.genHeader();
    code += metaGraph.genKernelFunc();
    code += metaGraph.genInvokeFunc();
    return code;
}
