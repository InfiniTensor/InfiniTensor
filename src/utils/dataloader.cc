#include "utils/dataloader.h"
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef TENSOR_PROTOBUF
#include "data.pb.h"
#endif
#include <fstream>

namespace infini {

void saveTensorData(TensorObj *tensor, std::string file_path) {
#ifdef TENSOR_PROTOBUF
    data::Tensor temp;
    temp.set_id("tensor_id");
    for (size_t i = 0; i < tensor->getDims().size(); ++i) {
        temp.add_shape(tensor->getDims()[i]);
    }
    temp.set_layout(data::LAYOUT_NHWC);
    if (tensor->getDType() == DataType::Float32) {
        temp.set_dtype(data::DTYPE_FLOAT);
        for (size_t i = 0; i < tensor->size(); ++i) {
            temp.add_data_float((tensor->getDataBlob()->getPtr<float *>())[i]);
        }
    } else if (tensor->getDType() == DataType::UInt32) {
        temp.set_dtype(data::DTYPE_UINT32);
        for (size_t i = 0; i < tensor->size(); ++i) {
            temp.add_data_uint32(
                (tensor->getDataBlob()->getPtr<uint32_t *>())[i]);
        }
    } else {
        IT_TODO_HALT();
    }

    std::ofstream fileout(file_path,
                          std::ios::out | std::ios::trunc | std::ios::binary);
    bool flag = temp.SerializeToOstream(&fileout);
    if (!flag) {
        std::cout << "Failed to write file " + file_path << std::endl;
    }
    fileout.close();
#else
    std::cout << "If you want to use this feature, please turn on USE_PROTOBUF "
                 "option in the cmake file."
              << std::endl;
#endif
}

void loadTensorData(TensorObj *tensor, std::string file_path) {
#ifdef TENSOR_PROTOBUF
    data::Tensor temp;
    std::ifstream filein(file_path, std::ios::in | std::ios::binary);
    bool flag = temp.ParseFromIstream(&filein);
    if (!flag) {
        std::cout << "Failed to read file " + file_path << std::endl;
    }

    if (tensor->getDType() == DataType::Float32) {
        std::vector<float> data_temp;
        for (int i = 0; i < temp.data_float_size(); ++i) {
            data_temp.push_back(temp.data_float(i));
        }
        tensor->copyin(data_temp);
    } else if (tensor->getDType() == DataType::UInt32) {
        std::vector<uint32_t> data_temp;
        for (int i = 0; i < temp.data_uint32_size(); ++i) {
            data_temp.push_back(temp.data_uint32(i));
        }
        tensor->copyin(data_temp);
    } else {
        IT_TODO_HALT();
    }

    filein.close();
#else
    std::cout << "If you want to use this feature, please turn on USE_PROTOBUF "
                 "option in the cmake file."
              << std::endl;
#endif
}

}; // namespace infini
