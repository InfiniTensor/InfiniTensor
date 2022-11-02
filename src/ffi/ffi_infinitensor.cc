#include <pybind11/stl.h>
#ifdef USE_CUDA
#include "cuda/operator_timer.h"
#endif
namespace py = pybind11;

namespace infini {

using namespace py::literals;
using policy = py::return_value_policy;

void register_operator_timer(py::module &m) {
#ifdef USE_CUDA
    using namespace opTimer;
    m.def("getPerfConvCudnn", &getPerfConvCudnn);
    m.def("getPerfConvBiasActCudnn", &getPerfConvBiasActCudnn);
    m.def("getPerfConvTransposed2dCudnn", &getPerfConvTransposed2dCudnn);
    m.def("getPerfMatmulCublas", &getPerfMatmulCublas);
    m.def("getPerfMatmulCublas", &getPerfMatmulCublas);
#endif
}

} // namespace infini

PYBIND11_MODULE(pyinfinitensor, m) { infini::register_operator_timer(m); }