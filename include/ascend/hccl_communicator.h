#pragma once
#include "core/communicator.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define ACLCHECK(ret)                                                          \
    do {                                                                       \
        assert(ret == ACL_SUCCESS);                                            \
    } while (0)
#define HCCLCHECK(ret)                                                         \
    do {                                                                       \
        assert(ret == HCCL_SUCCESS);                                           \
    } while (0)

namespace infini {

class HcclCommunicatorObj final : public CommunicatorObj {
  private:
    HcclComm comm;

  public:
    HcclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        const std::string filePath("./" + name + "_hccl_id.bin");
        int devId = rank;
        int devCount = worldSize;
        // 在 rootRank 获取 rootInfo
        HcclRootInfo rootInfo;
        int32_t rootRank = 0;

        if (devId == rootRank) {
            HCCLCHECK(HcclGetRootInfo(&rootInfo));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *)&rootInfo, sizeof(HcclRootInfo));
        } else {
            auto begin = std::chrono::steady_clock::now();
            while (!std::filesystem::exists(filePath)) {
                auto now = std::chrono::steady_clock::now();
                _IT_ASSERT_2(now < begin + std::chrono::seconds(10),
                             "time limit (10s) exceeded.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *)&rootInfo, sizeof(HcclRootInfo));
        }

        auto ret = HcclCommInitRootInfo(uint32_t(devCount), &rootInfo,
                                        uint32_t(devId), &comm);

        assert(ret == HCCL_SUCCESS);

        if (rank == 0) {
            std::filesystem::remove(filePath);
        }
    }

    // Get the actual ncclComm_t
    HcclComm getHcclComm() { return comm; }

    ~HcclCommunicatorObj() final {
        auto ret = HcclCommDestroy(comm);
        assert(ret == HCCL_SUCCESS);
    }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "HCCL communicator";
        return oss.str();
    }
};

} // namespace infini
