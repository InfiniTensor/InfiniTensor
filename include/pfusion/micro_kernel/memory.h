#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class MemoryOp : public MicroOp {
  private:
    size_t num, width;

  public:
    MemoryOp(const OpType _opType, const std::shared_ptr<Pointer> _src,
             const std::shared_ptr<Pointer> _dst, const size_t _num,
             const size_t _width, const std::vector<size_t> &_cond)
        : num(_num), width(_width) {
        opType = _opType;
        ptrs = {_src, _dst};
        cond = _cond;
    }
    // bool checkValid() override;
    ~MemoryOp() {}

    static inline std::shared_ptr<MicroOp>
    build(const OpType opType, const std::shared_ptr<Pointer> src,
          const std::shared_ptr<Pointer> dst, const size_t num,
          const size_t width) {
        return std::make_shared<MemoryOp>(opType, src, dst, num, width,
                                          std::vector<size_t>({}));
    }

    static inline std::shared_ptr<MicroOp>
    build(const OpType opType, const std::shared_ptr<Pointer> src,
          const std::shared_ptr<Pointer> dst, const size_t num,
          const size_t width, const std::vector<size_t> &cond) {
        return std::make_shared<MemoryOp>(opType, src, dst, num, width, cond);
    }

    std::shared_ptr<Pointer> getSrc() { return ptrs[0]; }
    std::shared_ptr<Pointer> getDst() { return ptrs[1]; }
    std::string generate() override;
    std::string generateWithCond();
    inline void print() override {
        if (opType == READ) {
            std::cout << id << " " << getName(opType) << " "
                      << getName(getSrc()->getType()) << " "
                      << getSrc()->getHash() << std::endl;
        } else if (opType == WRITE) {
            std::cout << id << " " << getName(opType) << " "
                      << getName(getDst()->getType()) << " "
                      << getDst()->getHash() << std::endl;
        } else {
            IT_ASSERT(false);
        }
    }
};
} // namespace memb
