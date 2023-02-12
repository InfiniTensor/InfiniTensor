#include <filesystem>
#include <pybind11/embed.h>
#include <test.h>

TEST(Python, pybind) {
    namespace fs = std::filesystem;
    namespace py = pybind11;
    using mod = py::module;

    py::scoped_interpreter _python;

    auto sys_path_append = mod::import("sys").attr("path").attr("append");
    sys_path_append(fs::path(__FILE__).parent_path().c_str());
    auto ans = mod::import("python").attr("inc")(1);
    EXPECT_EQ(ans.cast<int>(), 2);
}
