#pragma once
#include "core/communicator.h"
#include "xpu/bkcl.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#define checkXcclError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (BKCL_SUCCESS != err) {                                             \
            fprintf(stderr, "XCCL error in %s:%i.\n", __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

class XcclCommunicatorObj final : public CommunicatorObj {
  private:
    BKCLContext_t comm;

  public:
    XcclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        const std::string filePath("./" + name + "_xccl_id.bin");
        BKCLUniqueId commId;
        if (rank == 0) {
            checkXcclError(bkcl_get_unique_id(&commId));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *)&commId, sizeof(BKCLUniqueId));
        } else {
            auto begin = std::chrono::steady_clock::now();
            while (!std::filesystem::exists(filePath)) {
                auto now = std::chrono::steady_clock::now();
                _IT_ASSERT_2(now < begin + std::chrono::seconds(100),
                             "time limit (100s) exceeded.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *)&commId, sizeof(BKCLUniqueId));
        }
        checkXcclError(bkcl_init_rank(&comm, rank, worldSize, &commId));
        if (rank == 0) {
            std::filesystem::remove(filePath);
        }
    }

    BKCLContext_t getXcclComm() { return comm; }

    ~XcclCommunicatorObj() final { checkXcclError(bkcl_destroy_context(comm)); }
    virtual string toString() const final {
        std::ostringstream oss;
        oss << "XCCL communicator";
        return oss.str();
    }
};

} // namespace infini
