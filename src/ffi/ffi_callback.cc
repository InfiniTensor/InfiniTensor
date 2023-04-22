#include "core/graph.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infini {

namespace callback {

using namespace py::literals;

static std::function<void(const Graph &, string)> exportONNXImpl;
void exportONNX(const Graph &graph, const string &path) {
    IT_ASSERT(Py_IsInitialized(), "Python interpreter is not running.");
    static auto exportONNXImpl =
        py::module_::import("pyinfinitensor.onnx").attr("save_onnx");
    exportONNXImpl(graph, path);
}

} // namespace callback

} // namespace infini