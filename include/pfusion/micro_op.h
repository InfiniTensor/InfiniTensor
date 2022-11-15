#pragma once

#include "pfusion/common.h"
#include "pfusion/pointer.h"

namespace memb {

class MicroOp {
  protected:
    size_t id;
    OpType opType;
    std::vector<std::shared_ptr<Pointer>> ptrs;
    std::vector<size_t> cond;

  public:
    MicroOp() : opType(NONE), cond(0) {
        static int microOpId = 0;
        id = microOpId++;
    }
    virtual ~MicroOp() {}

    inline OpType getType() { return opType; }
    inline bool isMemoryOp() { return opType == READ || opType == WRITE; }
    inline std::vector<std::shared_ptr<Pointer>> getPtrs() { return ptrs; }

    // virtual bool checkValid() = 0;
    virtual std::string generate() = 0;
    virtual void print() = 0;
    static std::shared_ptr<MicroOp> merge(std::shared_ptr<MicroOp> op0,
                                          std::shared_ptr<MicroOp> op1);
};

class MicroGraph {
  public:
    MicroGraph() {}
    ~MicroGraph() {}

  private:
    std::vector<std::shared_ptr<MicroOp>> microOps;
    std::vector<std::pair<int, int>> edges;
};

} // namespace memb
