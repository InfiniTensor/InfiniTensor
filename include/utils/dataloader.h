#pragma once
#include "core/runtime.h"
#include "core/perf_engine.h"
#include <string>

namespace infini {

void loadTensorData(TensorObj *tensor, std::string file_path);
void saveTensorData(TensorObj *tensor, std::string file_path);
void savePerfEngineData(PerfEngine perfEngine, std::string file_path);
void loadPerfEngineData(PerfEngine perfEngine, std::string file_path);
} // namespace infini
