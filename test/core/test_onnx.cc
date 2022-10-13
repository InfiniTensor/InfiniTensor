#include "core/graph_factory.h"
#include "ffi/ffi_embed.h"
#include "test.h"

namespace py = pybind11;

namespace infini {

TEST(Onnx, import) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    // GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
    // const char *net = "/home/pairshoe/InfiniTensor/test/core/input.onnx";
    try {
        py::module::import("infinitensor"); // .attr("import_onnx")(gf, net);
        // py::module::import("infinitensor").attr("import_onnx")(gf, net);
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_ImportError)) {
            std::cerr << "Import Error. Don't forget to set environment "
                         "variable PYTHONPATH to contain "
                         "<repo-root>/python"
                      << std::endl;
        }
        throw;
    }
}

} // namespace infini
