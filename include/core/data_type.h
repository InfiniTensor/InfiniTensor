#include "core/common.h"

namespace infini {

class DataType {
  public:
    // legacy
    // These are just aligned with the type and index of onnx:
    // <https://onnx.ai/onnx/intro/concepts.html#element-type>
    static const DataType Float32;
    static const DataType UInt32;
    static const DataType UInt8;
    static const DataType Int8;
    static const DataType UInt16;
    static const DataType Int16;
    static const DataType Int32;
    static const DataType Int64;
    static const DataType Bool;
    static const DataType Float16;
    static constexpr size_t sizePerElement[]{
        sizeof(float),    sizeof(uint32_t), sizeof(uint8_t), sizeof(int8_t),
        sizeof(uint16_t), sizeof(int16_t),  sizeof(int32_t), sizeof(int64_t),
        sizeof(int8_t),   sizeof(uint8_t)};

    static constexpr std::string_view names[]{
        "Float32", "UInt32", "UInt8", "Int8", "UInt16",
        "Int16",   "Int32",  "Int64", "Bool", "Float16"};

    static constexpr std::string_view sizetostring[]{
        "float",   "uint32_t", "uint8_t", "int8_t", "uint16_t",
        "int16_t", "int32_t",  "int64_t", "int8_t", "uint8_t"};

  private:
    int index;

  public:
    // FIXME: default ctor should be deleted but json requires it. Solution:
    // https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
    DataType() = default;
    constexpr DataType(int index) : index(index) {}
    bool operator==(const DataType &rhs) const { return index == rhs.index; }
    bool operator<(const DataType &rhs) const { return index < rhs.index; }

    template <typename T> static std::string get() {
        IT_TODO_HALT_MSG("Unsupported data type");
    }
    size_t getSize() const { return sizePerElement[index]; }
    string toString() const { return string(names[index]); }
    string sizetoString() const { return string(sizetostring[index]); }
};

inline const DataType DataType::Float32(0);
inline const DataType DataType::UInt32(1);
inline const DataType DataType::UInt8(2), DataType::Int8(3),
    DataType::UInt16(4), DataType::Int16(5), DataType::Int32(6),
    DataType::Int64(7), DataType::Bool(8), DataType::Float16(9);
// Method definitions are out of the declaration due to GCC bug:
// https://stackoverflow.com/questions/49707184/explicit-specialization-in-non-namespace-scope-does-not-compile-in-gcc
template <> inline std::string DataType::get<float>() { return "float"; }
template <> inline std::string DataType::get<uint32_t>() { return "uint32_t"; }
template <> inline std::string DataType::get<uint8_t>() { return "uint8_t"; }
template <> inline std::string DataType::get<int8_t>() { return "int8_t"; }
template <> inline std::string DataType::get<uint16_t>() { return "uint16_t"; }
template <> inline std::string DataType::get<int16_t>() { return "int16_t"; }
template <> inline std::string DataType::get<int32_t>() { return "int32_t"; }
template <> inline std::string DataType::get<int64_t>() { return "int64_t"; }

template <int index> struct DT {};
template <> struct DT<0> {
    using t = float;
};
template <> struct DT<1> {
    using t = uint32_t;
};
template <> struct DT<2> {
    using t = uint8_t;
};
template <> struct DT<3> {
    using t = int8_t;
};
template <> struct DT<4> {
    using t = uint16_t;
};
template <> struct DT<5> {
    using t = int16_t;
};
template <> struct DT<6> {
    using t = int32_t;
};
template <> struct DT<7> {
    using t = int64_t;
};
template <> struct DT<8> {
    using t = int8_t;
};
template <> struct DT<9> {
    using t = uint16_t;
};

} // namespace infini
