#pragma once
#include "core/runtime.h"
#include <string>

namespace infini {

void loadTensorData(TensorObj *tensor, std::string file_path);
void saveTensorData(TensorObj *tensor, std::string file_path);
} // namespace infini
