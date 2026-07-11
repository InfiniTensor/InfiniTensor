#include "communication/infiniccl_communicator.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace infini {

void checkInfiniCcl(infinicclResult_t result, const string &operation) {
    IT_ASSERT(result == infinicclSuccess,
              operation + " failed with InfiniCCL status " +
                  std::to_string(static_cast<int>(result)));
}

InfiniCclCommunicatorObj::InfiniCclCommunicatorObj(const string &name,
                                                   int worldSize, int rank)
    : CommunicatorObj(worldSize, rank),
      uniqueIdPath("./" + name + "_infiniccl_id.bin"),
      ownsUniqueIdPath(rank == 0) {
    infinicclUniqueId uniqueId{};
    if (rank == 0) {
        checkInfiniCcl(infinicclGetUniqueId(&uniqueId),
                       "infinicclGetUniqueId");
        std::ofstream stream(uniqueIdPath, std::ios::binary | std::ios::trunc);
        IT_ASSERT(stream.good(), "Unable to create InfiniCCL unique-id file");
        stream.write(reinterpret_cast<const char *>(&uniqueId), sizeof(uniqueId));
        IT_ASSERT(stream.good(), "Unable to write InfiniCCL unique-id file");
    } else {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(10);
        while (!std::filesystem::exists(uniqueIdPath)) {
            IT_ASSERT(std::chrono::steady_clock::now() < deadline,
                      "Timed out waiting for InfiniCCL unique-id file");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::ifstream stream(uniqueIdPath, std::ios::binary);
        IT_ASSERT(stream.good(), "Unable to open InfiniCCL unique-id file");
        stream.read(reinterpret_cast<char *>(&uniqueId), sizeof(uniqueId));
        IT_ASSERT(stream.good(), "Unable to read InfiniCCL unique-id file");
    }
    checkInfiniCcl(infinicclCommInitRank(&comm, worldSize, uniqueId, rank),
                   "infinicclCommInitRank");
}

InfiniCclCommunicatorObj::~InfiniCclCommunicatorObj() {
    if (comm != nullptr) {
        (void)infinicclCommDestroy(comm);
    }
    if (ownsUniqueIdPath) {
        std::error_code error;
        std::filesystem::remove(uniqueIdPath, error);
    }
}

infinicclDataType_t toInfiniCclDataType(const DataType &dtype) {
    if (dtype == DataType::Int8) return infinicclInt8;
    if (dtype == DataType::Int16) return infinicclInt16;
    if (dtype == DataType::Int32) return infinicclInt32;
    if (dtype == DataType::Int64) return infinicclInt64;
    if (dtype == DataType::UInt8) return infinicclUInt8;
    if (dtype == DataType::UInt16) return infinicclUInt16;
    if (dtype == DataType::UInt32) return infinicclUInt32;
    if (dtype == DataType::UInt64) return infinicclUInt64;
    if (dtype == DataType::Float16) return infinicclFloat16;
    if (dtype == DataType::BFloat16) return infinicclBFloat16;
    if (dtype == DataType::Float32) return infinicclFloat32;
    if (dtype == DataType::Double) return infinicclFloat64;
    IT_TODO_HALT_MSG("InfiniCCL does not support this InfiniTensor dtype");
}

} // namespace infini
