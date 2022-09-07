#pragma once
#include <string>
#include "core/runtime.h"

namespace infini {
    
    void loadTensorData(TensorObj* tensor, std::string file_path);
    void saveTensorData(TensorObj* tensor, std::string file_path);


} // namespace infini
