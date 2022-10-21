#include "pfusion/memory_codegen.h"
#include "core/graph.h"
#include "operators/transpose.h"
#include "pfusion/common.h"
#include "pfusion/instantiate.h"

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
}

memb::MetaGraph instantiateGraph(infini::Graph graph) {
    memb::MetaGraph metaGraph;
    for (auto op : graph->getOperators()) {
        switch (op->getOpType()) {
        case infini::OpType::Abs:
            metaGraph.addNode(
                memb::instantiateAbs(op->getInputs()[0]->getDims()));
            break;
        case infini::OpType::Relu:
            metaGraph.addNode(
                memb::instantiateRelu(op->getInputs()[0]->getDims()));
            break;
        case infini::OpType::Transpose:
            metaGraph.addNode(memb::instantiateTranspose(
                op->getInputs()[0]->getDims(),
                infini::as<infini::TransposeObj>(op)->getPerm()));
            break;
        default:
            IT_ASSERT(false);
        }
    }
    return metaGraph;
}

std::string infini::MemoryCodegen::generate(Graph graph) {
    memb::MetaGraph metaGraph = instantiateGraph(graph);
    metaGraph.print();
    return "";
}
