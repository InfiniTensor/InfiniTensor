# InfiniTensor

[中文介绍](README_CN.md) | [Documentation](docs/INDEX.md)

[![Build](https://github.com/InfiniTensor/InfiniTensor/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/InfiniTensor/InfiniTensor/actions)
[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniTensor)](https://github.com/InfiniTensor/InfiniTensor/issues)
![License](https://img.shields.io/github/license/InfiniTensor/InfiniTensor)

InfiniTensor is a computation graph inference engine with ONNX import, graph transformations, and C++/Python execution APIs. It provides a native CPU runtime and single-device accelerator execution through InfiniOps and InfiniRT.

## Capabilities

- Import ONNX models, supply inputs, execute graphs, read outputs, and export graphs to ONNX.
- Build graphs directly with InfiniOps operators through bindings generated from the public SDK. Operators using supported parameter types need no dedicated backend wrapper.
- Prepare and reuse operator instances and workspaces. Use InfiniRT graph capture and replay when supported by the runtime and selected operator implementations.

InfiniTensor owns frontend semantics, graphs, and scheduling. InfiniOps supplies operator implementations; InfiniRT manages devices, memory, streams, and device graphs. Model compatibility depends on importer coverage and the operators, dtypes, and shapes available in the selected backend.

## Installation

Initialize the submodules after obtaining the source:

```bash
git submodule update --init --recursive
```

For the native CPU runtime:

```bash
make install-python INFINI=OFF PYTHON="$(command -v python3)"
```

For an InfiniOps backend, install compatible InfiniRT and InfiniOps builds first:

```bash
make install-python INFINI=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix \
  CHECK_BACKEND=infini \
  PYTHON="$(command -v python3)"
```

Use `ATEN=OFF` with native providers. ATen-backed providers require `ATEN=ON` and a matching Python/PyTorch environment.

## Documentation and development

- [Installation guide](docs/INSTALL_GUIDE_CN.md): dependencies, builds, provider configuration, and installation checks.
- [User guide](docs/USER_GUIDE_CN.md): device selection, ONNX inference, direct graph construction, and replay.
- [Developer guide](docs/DEVELOPMENT_CN.md): component responsibilities, extensions, tests, and CI.

The repository also includes tensor program and graph optimization research, including [EinNet](https://github.com/InfiniTensor/InfiniTensor/tree/NNET_e2e) and [PET](https://github.com/thu-pacman/PET).

Contributions use GitHub pull requests. Run relevant tests and clang-format 14, explain the resulting behavior, and include test output in the description. Obtain a reviewer approval before merging.

## Reference

Please cite EinNet or PET if they support your research:

```plaintext
@article{zheng2023einnet,
  title={EINNET: Optimizing Tensor Programs with Derivation-Based Transformations},
  author={Zheng, Liyan and Wang, Haojie and Zhai, Jidong and Hu, Muyan and Ma, Zixuan and Wang, Tuowei and Huang, Shuhong and Miao, Xupeng and Tang, Shizhi and Huang, Kezhao and Jia, Zhihao},
  booktitle={17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)},
  pages={739--755},
  year={2023}
}

@inproceedings{wang2021pet,
  title={PET: Optimizing tensor programs with partially equivalent transformations and automated corrections},
  author={Wang, Haojie and Zhai, Jidong and Gao, Mingyu and Ma, Zixuan and Tang, Shizhi and Zheng, Liyan and Li, Yuanzhi and Rong, Kaiyuan and Chen, Yuanyong and Jia, Zhihao},
  booktitle={15th USENIX Symposium on Operating Systems Design and Implementation (OSDI 21)},
  pages={37--54},
  year={2021}
}
```
