ffi: Foreign Function Interface 外部功能接口

ffi_embed.cc
// 这个文件定义了 start_interpreter 函数，该函数使用 pybind11 的 scoped_interpreter 初始化 Python 解释器。
// 该函数通过使用一个指向 scoped_interpreter 对象的静态指针来确保解释器只被初始化一次。这允许 InfiniTensor 库在需要时安全地调用 Python 代码。

ffi_infinitensor.cc
// 这个文件使用 pybind11 将 InfiniTensor 中的一些核心类和函数暴露给 Python。
// 它定义了 OpType 和 ActType 枚举，以及一些函数来获取操作的属性和运行时对象。这些函数将被 Python 代码调用，以便在 Python 中使用 InfiniTensor 的功能。