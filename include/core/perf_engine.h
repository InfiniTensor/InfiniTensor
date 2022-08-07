#pragma once
#include "core/graph.h"
#include "core/kernel.h"

namespace infini {

class PerfEngine {
  public:
    using Key = std::pair<KernelAttrs, OpAttrs>;

  private:
    map<Key, PerfRecord> data;

  public:
    static PerfEngine &getInstance() {
        static PerfEngine instance;
        return instance;
    }

    std::optional<PerfRecord> getPerfData(const Key &key) {
        auto it = data.find(key);
        if (it != data.end()) // find previous evaluating results
            return data.at(key);
        else
            return std::nullopt;
    }

    void setPerfData(const Key &key, const PerfRecord &record) {
        IT_ASSERT(data.find(key) == data.end(), "Perf data already exist");
        data.emplace(key, record);
    }
};

} // namespace infini