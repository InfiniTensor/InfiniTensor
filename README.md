# InfiniTensor

## Compilation on Lotus

``` bash
# Enter the root of InfiniTensor
source test/script/env_lotus.sh
mkdir build && cd build
cmake -DUSE_CUDA=ON .. && make -j 12
```

### CMake Options

There are several configurable CMake options, see the [CMakeLists.txt file](/CMakeLists.txt#L5).

- If `USE_BACKTRACE` is `ON`, `libdw-dev` have to be installed. See the README of [backward-cpp](https://github.com/bombela/backward-cpp) for details.
- If `USE_PROTOBUF` is `ON`, `protobuf` have to be installed. See the README of [protobuf](https://github.com/protocolbuffers/protobuf) for details.

## Contributor Guide

InfiniTensor development is based on the pull request on Github. Before requesting for merging, a PR should satisfy the following requirements

1. Pass all tests.
    1. Now CI on Github will test everything that can be tested in the ci environment, including code format. So, script `test/script/clang_format_inplace.sh` is for formatting all code.
    2. Contributors should run `ctest` manually and copy its output to the PR. Use fenced code blocks (triple backquotes, i.e., `` ``` ``) to avoid referencing in Github. Otherwise, `#` in the output is interpreted as a Github reference. Do not directly paste the ctest output in commit messages either for the same reason.
2. Receive at least one approval from reviewers.
3. PR title should be concise since it is going to be the commit message in the main branch after merging and squashing.

## Dependencies

- [backward-cpp](https://github.com/bombela/backward-cpp): [v1.6](https://github.com/bombela/backward-cpp/releases/tag/v1.6)
- [googletest](https://github.com/google/googletest): [v1.13.0](https://github.com/google/googletest/releases/tag/v1.13.0)
- [nlohmann_json_cmake_fetchcontent](https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent): [v3.10.5](https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent/releases/tag/v3.10.5)
- [pybind11](https://github.com/pybind/pybind11): [v2.10.3](https://github.com/pybind/pybind11/releases/tag/v2.10.3)
