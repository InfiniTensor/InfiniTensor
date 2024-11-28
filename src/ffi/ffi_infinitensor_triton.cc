#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <iostream>
#include <pybind11/embed.h>
#include "ffi/ffi_infinitensor_triton.h"
namespace py = pybind11;

void print_test_add() {
    // 创建一个 scoped_interpreter 对象，构造时初始化 Python 解释器，析构时清理
    py::scoped_interpreter guard{};

    // 获取当前工作目录
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        // 构建 Python 模块的路径
        std::string op_module_path = std::string(cwd) + "/python/tritonOp";
        std::cout << "python_module_path: " << op_module_path << std::endl;

        std::string python_module_path = "/home/jymiracle2204/.local/lib/python3.10/site-packages"; // 直接使用绝对路径
        std::cout << "python_module_path: " << python_module_path << std::endl;

        // 导入 sys 模块并添加模块路径
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, py::str(op_module_path));
        sys.attr("path").attr("insert")(0, py::str(python_module_path));



        // 导入 vector_add 模块
        py::module_ main_module = py::module_::import("vector_add");
        py::object vector_add = main_module.attr("vector_add");

        // 创建输入数据
        py::list a_list;
        py::list b_list;
        for (int i = 1; i <= 3; ++i) {
            a_list.append(i);
        }
        for (int i = 4; i <= 6; ++i) {
            b_list.append(i);
        }

        // 调用 Python 的 vector_add 函数
        py::object result = vector_add(a_list, b_list);

        std::cout << "Result: " << result << std::endl;
    } else {
        std::cerr << "Error getting current working directory: " << strerror(errno) << std::endl;
    }
}