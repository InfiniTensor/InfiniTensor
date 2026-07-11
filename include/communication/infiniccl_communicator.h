#pragma once

#include "core/communicator.h"
#include "core/data_type.h"

#include <comm.h>

namespace infini {

class InfiniCclCommunicatorObj final : public CommunicatorObj {
  public:
    InfiniCclCommunicatorObj(const string &name, int worldSize, int rank);
    ~InfiniCclCommunicatorObj() final;

    infinicclComm_t getComm() const { return comm; }

  private:
    infinicclComm_t comm = nullptr;
    string uniqueIdPath;
    bool ownsUniqueIdPath = false;
};

void checkInfiniCcl(infinicclResult_t result, const string &operation);
infinicclDataType_t toInfiniCclDataType(const DataType &dtype);

} // namespace infini
