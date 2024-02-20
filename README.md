# InfiniTensor

[中文项目简介](/README_CN.md) | Documentation | [中文文档](/docs/INDEX.md)

[![Build](https://github.com/InfiniTensor/InfiniTensor/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/InfiniTensor/InfiniTensor/actions)
[![issue](https://img.shields.io/github/issues/InfiniTensor/InfiniTensor)](https://github.com/InfiniTensor/InfiniTensor/issues)
![license](https://img.shields.io/github/license/InfiniTensor/InfiniTensor)

InfiniTensor is a high-performance inference engine tailored for GPUs and AI accelerators. Its design focuses on effective deployment and swift academic validation.

## Get started

### Make Commands

- `make`/`make build`: Builds the project;
- `make install-python`: Builds the project then install the python frontend;
- `make test-cpp`: Builds the project then run cpp unit tests;
- `make test-onnx`: Run python unit tests;

---

> - Sets env: `TEST=OFF` to accelerate compiling.
> - Sets env: `CUDA=ON` to enable cuda.
> - Sets env: `BANG=ON` to enable bang.

### CMake Options

There are several configurable CMake options, see the [CMakeLists.txt](/CMakeLists.txt#L5) file.

- If `USE_BACKTRACE` is `ON`, `libdw-dev` have to be installed. See the README of [backward-cpp](https://github.com/bombela/backward-cpp) for details.
- If `USE_PROTOBUF` is `ON`, `protobuf` have to be installed. See the README of [protobuf](https://github.com/protocolbuffers/protobuf) for details.
- If `USE_CUDA` is `ON`, `cuda` have to be installed.

## Roadmap

- [RefactorGraph](https://github.com/InfiniTensor/RefactorGraph) is a newly designed AI framework that is set to replace the current main branch.
- [EinNet](https://github.com/InfiniTensor/InfiniTensor/tree/NNET_e2e) is going to be merged into the main branch.
- Integration of [PET](https://github.com/thu-pacman/PET), a tensor program optimizer supporting partially equivalent transformations.
- Supported hardware
  - ✔ NVIDIA GPU
  - ✔ Cambricon MLU
  - ✔ Kunlunxin XPU
  - ⬜ Ascend NPU

## Contributor Guide

InfiniTensor development is based on the pull request on Github. Before requesting for merging, a PR should satisfy the following requirements

1. Pass all tests.
    1. Now CI on Github will test everything that can be tested in the ci environment, including code format. So, script `test/script/clang_format_inplace.sh` is for formatting all code.
    2. Contributors should run `ctest` manually and copy its output to the PR. Use fenced code blocks (triple backquotes, i.e., `` ``` ``) to avoid referencing in Github. Otherwise, `#` in the output is interpreted as a Github reference. Do not directly paste the ctest output in commit messages either for the same reason.
2. Receive at least one approval from reviewers.
3. PR title should be concise since it is going to be the commit message in the main branch after merging and squashing.

## Reference

Please cite EinNet or PET in your publications if it helps your research:

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
