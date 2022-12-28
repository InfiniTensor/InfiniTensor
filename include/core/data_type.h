#include "core/common.h"

namespace infini {

class DataType {
  public:
    static const DataType Float32;
    static const DataType UInt32;
    static const DataType Int32;
    static constexpr size_t sizePerElement[]{sizeof(float), sizeof(uint32_t), sizeof(int32_t)};
    static constexpr std::string_view names[]{"Float32", "UInt32", "Int32"};

  private:
    int index;

  public:
    // FIXME: default ctor should be deleted but json requires it. Solution:
    // https://github.com/nlohmann/json#how-can-i-use-get-for-non-default-constructiblenon-copyable-types
    DataType() = default;
    constexpr DataType(int index) : index(index) {}
    bool operator==(const DataType &rhs) const { return index == rhs.index; }
    bool operator<(const DataType &rhs) const { return index < rhs.index; }

    template <typename T> static DataType get() {
        IT_TODO_HALT_MSG("Unsupported data type");
    }
    size_t getSize() const { return sizePerElement[index]; }
    string toString() const { return string(names[index]); }
};

inline const DataType DataType::Float32(0);
inline const DataType DataType::UInt32(1);
inline const DataType DataType::Int32(2);
// Method definitions are out of the declaration due to GCC bug:
// https://stackoverflow.com/questions/49707184/explicit-specialization-in-non-namespace-scope-does-not-compile-in-gcc
template <> inline DataType DataType::get<float>() { return Float32; }
template <> inline DataType DataType::get<uint32_t>() { return UInt32; }
template <> inline DataType DataType::get<int32_t>() { return Int32; }

} // namespace infini
