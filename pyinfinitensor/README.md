# pyinfinitensor

InfiniTensor 的 Python 前端。

- [ ] 从 ONNX 导入模型
- [ ] 从 Pytorch 导入模型
- [ ] 从 PaddlePaddle 导入模型
- [ ] 模型导出到 ONNX
- [ ] 模型导出到 Pytorch
- [ ] 模型导出到 PaddlePaddle

## python 工程结构及打包方法

本项目使用 [pyproject.toml] 文件定义，目录结构采用 [src 布局](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout)，后端库会被[打包](https://setuptools.pypa.io/en/latest/userguide/datafiles.html#package-data)，并支持[自动的依赖安装](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#declaring-required-dependency)。

参考 [setuptools 的文档](https://setuptools.pypa.io/en/latest/userguide/index.html)。
