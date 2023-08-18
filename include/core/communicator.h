#pragma once
#include "object.h"
#include "ref.h"

namespace infini {

// base class
class CommunicatorObj : public Object {
  protected:
    const int worldSize;
    const int rank;

  public:
    CommunicatorObj(int worldSize, int rank)
        : worldSize(worldSize), rank(rank) {}

    virtual ~CommunicatorObj() = default;
    virtual int getWorldSize() const { return worldSize; }
    virtual int getRank() const { return rank; }
};

} // namespace infini
