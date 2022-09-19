#include "core/perf_engine.h"
#include <fstream>
namespace infini {

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
    this->getInstance() = j;
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
    int x = p.toString() == "Float32" ? 0 : 1;
    j = x;
}
void from_json(const json &j, DataType &p) { p = DataType(j.get<int>()); }
void to_json(json &j, const PerfRecord &p) { p->to_json(j); }
void from_json(const json &j, PerfRecord &p) {
    int type = j["type"].get<int>();
    if (type == 1) {
        ConvCuDnnPerfRecordObj tmp;
        tmp.from_json(j);
        p = make_ref<ConvCuDnnPerfRecordObj>(tmp);
    } else if (type == 2) {
        MatmulCudnnPerfRecordObj tmp;
        tmp.from_json(j);
        p = make_ref<MatmulCudnnPerfRecordObj>(tmp);
    } else {
        p->from_json(j);
    }
}

void to_json(json &j, const PerfEngine &p) {
    PerfEngine t = p;
    j["data"] = t.get_data();
}
void from_json(const json &j, PerfEngine &p) {

    auto tmp = j["data"].get<map<PerfEngine::Key, PerfRecord>>();

    p.set_data(tmp);
}

} // namespace infini