# InfiniTensor

[中文介绍](/README_CN.md) | [中文文档](/docs/INDEX.md)

[![Build](https://github.com/InfiniTensor/InfiniTensor/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/InfiniTensor/InfiniTensor/actions)
[![issue](https://img.shields.io/github/issues/InfiniTensor/InfiniTensor)](https://github.com/InfiniTensor/InfiniTensor/issues)
![license](https://img.shields.io/github/license/InfiniTensor/InfiniTensor)
![star](https://atomgit.com/InfiniTensor/InfiniTensor/star/badge.svg)

InfiniTensor imports, transforms, and executes computation graphs. It provides a native CPU runtime and uses InfiniOps and InfiniRT for accelerator execution. Distributed execution is built on InfiniCCL.

Hardware-specific SDK headers and implementations stay outside InfiniTensor. The available accelerator devices and operators are determined by the installed InfiniOps and InfiniRT build.

## Quick start

Initialize the pinned lower-stack sources after cloning:

```bash
git submodule update --init --recursive
```

Install the Python package for the native CPU runtime:

```bash
make install-python INFINI=OFF PYTHON="$(command -v python3)"
```

To use an accelerator, first install matching InfiniOps and InfiniRT prefixes for the target machine, then build InfiniTensor against them:

```bash
make install-python \
  INFINI=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix \
  PYTHON="$(command -v python3)"
```

If the selected InfiniOps build requires Python provider modules, pass them with `PROVIDER_MODULES` during installation and `INFINIOPS_PROVIDER_MODULES` at runtime.

Common targets:

- `make build`: build the C++ project.
- `make install-python`: build and install `pyinfinitensor`.
- `make test-cpp`: run C++ tests.
- `make test-onnx`: run ONNX frontend tests.
- `make test-api`: run Python API tests.
- `make clean`: remove generated build files.

See the [installation guide](/docs/INSTALL_GUIDE_CN.md) for lower-stack preparation, offline installation, ABI matching, and post-install checks.

## Hardware and model compatibility

InfiniTensor does not maintain a separate static hardware matrix. Check the exact InfiniOps and InfiniRT build, vendor SDK, driver, and device runtime used on the target machine.

Model compatibility is not limited to a fixed model list. It depends on ONNX importer coverage, operator semantics, data types and shapes, and the implementations available in the selected InfiniOps backend. Unsupported operators fail explicitly instead of silently falling back to CPU.

## Build notes

- `TEST=OFF` skips test targets during compilation.
- `BACKTRACE=OFF` is the default for portable Release builds. Enable it only when `libdw-dev` is available.
- `USE_PROTOBUF=ON` requires a compatible Protobuf installation.
- `INFINIOPS_ROOT` and `INFINIRT_ROOT` must refer to matching lower-stack builds.
- ATen-backed InfiniOps builds must use the same Python/PyTorch C++11 ABI as InfiniTensor.
- `INFINIOPS_PROVIDER_LIBRARY_DIRS` is only needed when provider libraries are outside the selected Python/PyTorch environment.

## Documentation

- [Installation guide](/docs/INSTALL_GUIDE_CN.md)
- [User guide](/docs/USER_GUIDE_CN.md)
- [Distributed example](/examples/distributed/README.md)

## Roadmap

- [RefactorGraph](https://github.com/InfiniTensor/RefactorGraph) is the next-generation graph framework under development.
- [EinNet](https://github.com/InfiniTensor/InfiniTensor/tree/NNET_e2e) provides derivation-based tensor program optimization.
- [PET](https://github.com/thu-pacman/PET) provides partially equivalent transformations and automated corrections.
- Accelerator support continues to evolve through InfiniOps and InfiniRT.

## Contributor guide

Development uses GitHub pull requests. Before requesting review:

1. Run the relevant tests and formatting checks. Use `test/script/clang_format_inplace.sh` for C++ formatting.
2. Include the `ctest` result in the pull request when C++ code changes. Put terminal output in a fenced code block.
3. Use a concise pull request title because squash merges use it as the commit message.
4. Obtain at least one reviewer approval.

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
