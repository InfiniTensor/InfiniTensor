#pragma once
#include "core/common.h"

namespace infini {

class DataType {
  public:
    // <https://onnx.ai/onnx/intro/concepts.html#element-type>
    static const DataType Undefine;
    static const DataType Float32;
    static const DataType UInt8;
    static const DataType Int8;
    static const DataType UInt16;
    static const DataType Int16;
    static const DataType Int32;
    static const DataType Int64;
    static const DataType String;
    static const DataType Bool;
    static const DataType Float16;
    static const DataType Double;
    static const DataType UInt32;
    static const DataType UInt64;
    static const DataType BFloat16;
    // "sizePerElement" show the DType to cpu_type
    // DataType::Bool -> int8_t   DataType::Float16 -> uint16_t
    static constexpr size_t sizePerElement[]{0,
                                             sizeof(float),
                                             sizeof(uint8_t),
                                             sizeof(int8_t),
                                             sizeof(uint16_t),
                                             sizeof(int16_t),
                                             sizeof(int32_t),
                                             sizeof(int64_t),
                                             sizeof(std::string),
                                             sizeof(int8_t),
                                             sizeof(uint16_t),
                                             sizeof(double),
                                             sizeof(uint32_t),
                                             sizeof(uint64_t),
                                             0,
                                             0,
                                             sizeof(uint16_t)};

    static constexpr std::string_view names[]{
        "Undefine",    "Float32", "UInt8",  "Int8",   "UInt16",
        "Int16",       "Int32",   "Int64",  "String", "Bool",
        "Float16",     "Double",  "UInt32", "UInt64", "PlaceHolder",
        "PlaceHolder", "BFloat16"};

    static constexpr int cpuType[]{-1, 0, 2, 3, 4, 5,  6,  7, -1,
                                   3,  4, 9, 1, 8, -1, -1, 4};

  private:
    int index;

  public:
    // FIXME: default ctor should be deleted but json requires it. Solution:
    // https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
    DataType() = default;
    constexpr DataType(int index) : index(index) {}
    bool operator==(const DataType &rhs) const { return index == rhs.index; }
    bool operator<(const DataType &rhs) const { return index < rhs.index; }

    template <typename T> static int get() {
        IT_TODO_HALT_MSG("Unsupported data type");
    }
    size_t getSize() const { return sizePerElement[index]; }
    string toString() const { return string(names[index]); }
    int cpuTypeInt() const { return cpuType[index]; }
    int getIndex() const { return index; }
};

// Method definitions are out of the declaration due to GCC bug:
// https://stackoverflow.com/questions/49707184/explicit-specialization-in-non-namespace-scope-does-not-compile-in-gcc
template <> inline int DataType::get<float>() { return 0; }
template <> inline int DataType::get<uint32_t>() { return 1; }
template <> inline int DataType::get<uint8_t>() { return 2; }
template <> inline int DataType::get<int8_t>() { return 3; }
template <> inline int DataType::get<uint16_t>() { return 4; }
template <> inline int DataType::get<int16_t>() { return 5; }
template <> inline int DataType::get<int32_t>() { return 6; }
template <> inline int DataType::get<int64_t>() { return 7; }
template <> inline int DataType::get<uint64_t>() { return 8; }
template <> inline int DataType::get<double>() { return 9; }

template <int index> struct DT {};
template <> struct DT<0> { using t = bool; };
template <> struct DT<1> { using t = float; };
template <> struct DT<2> { using t = uint8_t; };
template <> struct DT<3> { using t = int8_t; };
template <> struct DT<4> { using t = uint16_t; };
template <> struct DT<5> { using t = int16_t; };
template <> struct DT<6> { using t = int32_t; };
template <> struct DT<7> { using t = int64_t; };
template <> struct DT<8> { using t = char; };
template <> struct DT<9> { using t = int8_t; };
template <> struct DT<10> { using t = uint16_t; };
template <> struct DT<11> { using t = double; };
template <> struct DT<12> { using t = uint32_t; };
template <> struct DT<13> { using t = uint64_t; };
template <> struct DT<16> { using t = uint16_t; };

} // namespace infini
