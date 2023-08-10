#pragma once
#include "core/communicator.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nccl.h>
#include <thread>

#define checkNcclError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (ncclSuccess != err) {                                              \
            fprintf(stderr, "NCCL error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    ncclGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

class NcclCommunicatorObj final : public CommunicatorObj {
  private:
    ncclComm_t comm;

  public:
    NcclCommunicatorObj(int worldSize, int rank) {
        const std::string filePath("./nccl_comm_id.temp");
        ncclUniqueId commId;
        if (rank == 0) {
            checkNcclError(ncclGetUniqueId(&commId));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *)&commId, sizeof(ncclUniqueId));

        } else {
            while (!std::filesystem::exists(filePath)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *)&commId, sizeof(ncclUniqueId));
        }
        checkNcclError(ncclCommInitRank(&comm, worldSize, commId, rank));
        std::filesystem::remove(filePath);
    }

    void finalize() { checkNcclError(ncclCommFinalize(comm)); }

    ~NcclCommunicatorObj() final {
        finalize();
        checkNcclError(ncclCommDestroy(comm));
    }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "NCCL communicator";
        return oss.str();
    }
};

} // namespace infini
