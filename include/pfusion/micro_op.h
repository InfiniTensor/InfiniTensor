#pragma once

#include "pfusion/common.h"
#include "pfusion/pointer.h"

namespace memb {

class MicroOp {
  public:
    enum MicroOpType {
        memory = 1,
        kernel,
    };

  protected:
    MicroOpType type;
    int id;

  public:
    MicroOp() {
        static int microOpId = 0;
        id = microOpId++;
    }
    virtual ~MicroOp() {}

    inline MicroOpType getType() { return type; }

    // virtual bool checkValid() = 0;
    virtual std::string generate() = 0;
    virtual void print() = 0;
};

class MicroGraph {
  public:
    MicroGraph() {}
    ~MicroGraph() {}

  private:
    std::vector<std::shared_ptr<MicroOp>> microOps;
    std::vector<std::pair<int, int>> deps;
};

} // namespace memb
