#pragma once
#include "core/common.h"
#include "ref.h"

namespace infini {

using UidBaseType = int;

class Guid {
  private:
    UidBaseType guid;

  private:
    UidBaseType generateGuid() {
        static UidBaseType guidCnt = 0;
        return ++guidCnt;
    }

  public:
    Guid() { guid = generateGuid(); }
    Guid(const Guid &rhs) { guid = generateGuid(); }
    Guid &operator=(const Guid &rhs) {
        guid = generateGuid();
        return *this;
    }

    operator UidBaseType() const { return guid; }
};

class Fuid {
  private:
    UidBaseType fuid;

  private:
    UidBaseType generateFuid() {
        static UidBaseType guidCnt = 0;
        return ++guidCnt;
    }

  public:
    Fuid() { fuid = generateFuid(); }

    operator UidBaseType() const { return fuid; }
};

class Object {
  protected:
    Guid guid;

  public:
    virtual ~Object(){};
    virtual string toString() const = 0;
    void print() { std::cout << toString() << std::endl; }
    UidBaseType getGuid() const { return guid; }
};

inline std::ostream &operator<<(std::ostream &os, const Object &obj) {
    os << obj.toString();
    return os;
}

// Overload for Ref-wrapped Object
template <typename T,
          typename std::enable_if_t<std::is_base_of_v<Object, T>> * = nullptr>
inline std::ostream &operator<<(std::ostream &os, const Ref<T> &obj) {
    os << obj->toString();
    return os;
}

} // namespace infini