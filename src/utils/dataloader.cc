#include "utils/dataloader.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "data.pb.h"
#include <fstream>

namespace infini {

void saveTensorData(TensorObj *tensor, std::string file_path) {
    data::Tensor temp;
    temp.set_id("tensor_id");
    temp.mutable_shape()->CopyFrom(
        {tensor->getDims().begin(), tensor->getDims().end()});
    temp.set_layout(data::LAYOUT_NHWC);
    temp.set_dtype(data::DTYPE_FLOAT);
    temp.mutable_data_float()->CopyFrom(
        {tensor->getDataBlob()->getPtr<float *>(),
         tensor->getDataBlob()->getPtr<float *>() + tensor->size()});
    std::ofstream fileout(file_path,
                          std::ios::out | std::ios::trunc | std::ios::binary);
    bool flag = temp.SerializeToOstream(&fileout);
    if (!flag) {
        std::cout << "Failed to write file " + file_path << std::endl;
    }
    fileout.close();
}

void loadTensorData(TensorObj *tensor, std::string file_path) {
    data::Tensor temp;
    std::vector<float> data_temp;
    std::ifstream filein(file_path, std::ios::in | std::ios::binary);
    bool flag = temp.ParseFromIstream(&filein);
    if (!flag) {
        std::cout << "Failed to read file " + file_path << std::endl;
    }

    for (int i = 0; i < temp.data_float_size(); ++i) {
        data_temp.push_back(temp.data_float(i));
    }

    tensor->copyData(data_temp);

    filein.close();
}

}; // namespace infini
