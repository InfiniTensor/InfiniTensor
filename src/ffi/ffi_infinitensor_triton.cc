#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <iostream>
#include <pybind11/embed.h>
namespace py = pybind11;

void print_test_add() {
    // 创建一个 scoped_interpreter 对象，构造时初始化 Python 解释器，析构时清理
    py::scoped_interpreter guard{};

    // 导入 vector_add 函数
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
}