#pragma once

#include "pfusion/common.h"

namespace memb {
class Pointer {
  private:
    const MemType memType;
    const std::string name, offset;

  public:
    Pointer(MemType _memType, std::string _name, std::string _offset)
        : memType(_memType), name(_name), offset(_offset) {}
    ~Pointer() {}

    static inline std::shared_ptr<Pointer> buildPtr(MemType memType,
                                                    std::string name) {
        return std::make_shared<Pointer>(memType, name, "0");
    }
    static inline std::shared_ptr<Pointer>
    buildPtr(MemType memType, std::string name, std::string offset) {
        return std::make_shared<Pointer>(memType, name, offset);
    }
    static inline std::shared_ptr<Pointer>
    buildPtr(std::shared_ptr<Pointer> ptr) {
        return std::make_shared<Pointer>(ptr->getType(), ptr->getName(),
                                         ptr->getOffset());
    }
    static inline std::shared_ptr<Pointer>
    buildPtr(std::shared_ptr<Pointer> ptr, std::string offset) {
        return std::make_shared<Pointer>(ptr->getType(), ptr->getName(),
                                         ptr->getOffset() + " + " + offset);
    }
    static inline std::shared_ptr<Pointer> buildPtrByTensorGuid(size_t guid) {
        return std::make_shared<Pointer>(
            MemType::DRAM, "tensor_ptr_" + std::to_string(guid), "0");
    }

    inline const MemType getType() { return memType; }
    inline const std::string getName() { return name; }
    inline const std::string getOffset() { return offset; }
    inline const std::string generate() { return name + "[" + offset + "]"; }
    inline bool equal(std::shared_ptr<Pointer> ptr) {
        if (name == ptr->getName() && offset == ptr->getOffset()) {
            IT_ASSERT(memType == ptr->getType());
            return true;
        }
        return false;
    }
    inline const size_t getHash() {
        std::hash<MemType> memTypeHash;
        std::hash<std::string> stringHash;
        size_t ret = memTypeHash(memType);
        ret = hashAppend(ret, stringHash(name));
        ret = hashAppend(ret, stringHash(offset));
        return ret;
    }
};

} // namespace memb