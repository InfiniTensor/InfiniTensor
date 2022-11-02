#pragma once

#include "pfusion/common.h"
#include "pfusion/micro_op.h"
#include "pfusion/pointer.h"

namespace memb {
class TensorMapping {
  private:
    std::vector<size_t> shape, map;
    std::string name;

  public:
    TensorMapping(const std::string _name, const std::vector<size_t> &_shape,
                  const std::vector<size_t> &_map) {
        name = "offset_" + _name;
        IT_ASSERT(_shape.size() > 0 && _shape.size() < 10);
        for (auto x : _shape) {
            shape.emplace_back(x);
        }
        IT_ASSERT(_map.size() > 0 && _map.size() < 10);
        for (auto x : _map) {
            map.emplace_back(x);
        }
    }
    ~TensorMapping() {}
    inline std::string offset() { return name; }
    inline size_t getHash() {
        std::hash<size_t> hasher;
        std::hash<std::string> stringHasher;
        size_t ret = stringHasher(name);
        ret = hashAppend(ret, hasher(shape.size()));
        for (auto x : shape) {
            ret = hashAppend(ret, hasher(x));
        }
        ret = hashAppend(ret, hasher(map.size()));
        for (auto x : map) {
            ret = hashAppend(ret, hasher(x));
        }
        return ret;
    }
    std::string genOffset();
};

class MetaOp {
  public:
    int id;
    int main_loop_st, main_loop_ed, numBlocks, numGroups, numReg, numSmem, numLanes;
    std::vector<std::shared_ptr<MicroOp>> microOps;
    std::vector<std::shared_ptr<Pointer>> ptrs;
    std::vector<std::shared_ptr<TensorMapping>> mappings;
    MetaOp() {
        static int metaOpId = 0;
        id = metaOpId++;
    }
    ~MetaOp() {}

    inline void setLoopSt(int _main_loop_st) { main_loop_st = _main_loop_st; }
    inline void setLoopEd(int _main_loop_ed) { main_loop_ed = _main_loop_ed; }
    inline int getLoopSt() { return main_loop_st; }
    inline int getLoopEd() { return main_loop_ed; }

    void optimize();
    std::string genKernelFunc();
    std::string genInvokeFunc();

    static std::shared_ptr<MetaOp> merge(std::shared_ptr<MetaOp> metaOp0,
                                         std::shared_ptr<MetaOp> metaOp1);

    inline void print() {
        std::cout << "MetaOp: " << id << std::endl;
        for (auto microOp : microOps) {
            microOp->print();
        }
    }
    bool checkValid() {
        // TODO: check valid
        return true;
    };
};

} // namespace memb