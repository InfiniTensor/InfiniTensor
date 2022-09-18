#pragma once
#include "core/graph.h"
#include "core/kernel.h"
namespace infini {

class PerfEngine {
  public:
    // TODO: Key should be OpPerfKey + Context(maybe implicat) to support
    // multiple candiate kernels.
<<<<<<< HEAD
    using Key = std::pair<KernelAttrs, OpPerfKey>;
    PerfEngine() = default;
    // PerfEngine is singleton
    PerfEngine(PerfEngine &other) = delete;
    PerfEngine &operator=(PerfEngine const &) = delete;
=======

    using Key = std::pair<KernelAttrs, OpPerfKey>;
>>>>>>> cf58b99 (clang format)

  private:
    map<Key, PerfRecord> data;

  public:
    static PerfEngine &getInstance() {
        static PerfEngine instance;
        return instance;
    }

    PerfRecord getPerfData(const Key &key) {
        auto it = data.find(key);
        if (it != data.end()) // find previous evaluating results
            return data.at(key);
        else
            return nullptr;
    }

    void setPerfData(const Key &key, PerfRecord record) {
        IT_ASSERT(data.find(key) == data.end(), "Perf data already exist");
        data.emplace(key, record);
    }
    map<Key, PerfRecord> get_data() { return data; }
    void set_data(map<Key, PerfRecord> data) { this->data = data; }
};
void to_json(json &j, const OpPerfKey &p);
void from_json(const json &j, OpPerfKey &p);
void to_json(json &j, const DataType &p);
void from_json(const json &j, DataType &p);
void to_json(json &j, const PerfRecord &p);
void from_json(const json &j, PerfRecord &p);
void to_json(json &j, PerfRecord *p);
void from_json(const json &j, PerfRecord *p);
void to_json(json &j, const PerfEngine &p);
void from_json(const json &j, PerfEngine &p);
} // namespace infini