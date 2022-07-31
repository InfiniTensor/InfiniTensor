#pragma once
#include "core/common.h"

namespace it {

using GuidBaseType = int;

class Guid {
  private:
    GuidBaseType guid;

  private:
    GuidBaseType generateGuid() {
        static GuidBaseType guidCnt = 0;
        return ++guidCnt;
    }

  public:
    Guid() { guid = generateGuid(); }
    Guid(const Guid &rhs) { guid = generateGuid(); }
    Guid &operator=(const Guid &rhs) {
        guid = generateGuid();
        return *this;
    }

    operator GuidBaseType() const { return guid; }
};

class Object {
  protected:
    Guid guid;

  public:
    virtual ~Object(){};
    virtual string toString() const = 0;
    void print() { std::cout << toString() << std::endl; }
    Guid getGuid() const { return guid; }
};

inline std::ostream &operator<<(std::ostream &os, const Object &obj) {
    os << obj.toString();
    return os;
}

}