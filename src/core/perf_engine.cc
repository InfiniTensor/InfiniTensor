#include "core/perf_engine.h"
#include <fstream>
namespace infini {

REGISTER_CONSTRUCTOR(0, PerfRecordObj::from_json);

void PerfEngine::savePerfEngineData(std::string file_path) {
    std::ofstream fileout(file_path,
                          std::ios::out | std::ios::trunc | std::ios::binary);
    json t = this->getInstance();
    fileout << t << std::endl;
    fileout.close();
}

void PerfEngine::loadPerfEngineData(std::string file_path) {
    std::ifstream filein(file_path, std::ios::in | std::ios::binary);
    string t;
    filein >> t;
    json j = json::parse(t);
    from_json(j, this->getInstance());
    filein.close();
}

/* json register should in the common namespace with corresponding type*/
void to_json(json &j, const OpPerfKey &p) {
    j = json{{"hashType", p.hash}, {"opType", p.opType}, {"attrs", p.attrs}};
}
void from_json(const json &j, OpPerfKey &p) {
    j.at("hashType").get_to(p.hash);
    j.at("opType").get_to(p.opType);
    j.at("attrs").get_to(p.attrs);
}
void to_json(json &j, const DataType &p) {
    j = p.toString() == "Float32" ? 0 : 1;
}
void from_json(const json &j, DataType &p) { p = DataType(j.get<int>()); }
void to_json(json &j, const PerfRecord &p) { p->to_json(j); }
void from_json(const json &j, PerfRecord &p) {
    int type = j["type"].get<int>();
    p = PerfRecordRegistry::getInstance().getConstructor(type)(j);
}

void to_json(json &j, const PerfEngine &p) {
    auto &x = p.getInstance();
    j["data"] = x.get_data();
}
void from_json(const json &j, PerfEngine &p) {
    auto tmp = j["data"].get<map<PerfEngine::Key, PerfRecord>>();
    p.set_data(tmp);
}

} // namespace infini