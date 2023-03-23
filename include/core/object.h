#pragma once
#include "core/common.h"
#include "ref.h"

namespace infini {

using UidBaseType = int;

class Uid {
  private:
    UidBaseType uid;

  public:
    Uid(UidBaseType uid) : uid(uid) {}
    Uid &operator=(const Uid &rhs) = delete;

    operator UidBaseType() const { return uid; }
};

class Guid : public Uid {
  private:
    UidBaseType generateGuid() {
        static UidBaseType guidCnt = 0;
        return ++guidCnt;
    }

  public:
    Guid() : Uid(generateGuid()) {}
    Guid(const Guid &rhs) : Uid(generateGuid()) {}
};

/**
 * @brief Family unique ID. Cloned tensors shared the same FUID.
 */
class Fuid : public Uid {
  private:
    UidBaseType generateFuid() {
        static UidBaseType fuidCnt = 0;
        return ++fuidCnt;
    }

  public:
    Fuid() : Uid(generateFuid()) {}
    Fuid(const Fuid &fuid) : Uid(fuid) {}
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
