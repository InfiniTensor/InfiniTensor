#pragma once
#include "core/graph.h"
#include "core/kernel.h"
#include <nlohmann/json_fwd.hpp>
using json = nlohmann::json;
namespace infini {

class PerfEngine {
  public:
    // TODO: Key should be OpPerfKey + Context(maybe implicat) to support
    // multiple candiate kernels.
    using Key = std::pair<KernelAttrs, OpPerfKey>;
    PerfEngine() = default;
    // PerfEngine is singleton
    PerfEngine(PerfEngine &other) = delete;
    PerfEngine &operator=(PerfEngine const &) = delete;

  private:
    map<Key, PerfRecord> data;

  public:
    static PerfEngine &getInstance() {
        static PerfEngine instance;
        return instance;
    }

    /**
     * @brief Get the Perf Data object
     *
     * @return PerfRecord nullptr if no record is fnoud.
     */
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
    void savePerfEngineData(std::string file_path);
    void loadPerfEngineData(std::string file_path);
};
void to_json(json &j, const PerfEngine &p);
void from_json(const json &j, PerfEngine &p);

} // namespace infini
