#pragma once
#include "object.h"
#include "ref.h"

namespace infini {

// base class
class CommunicatorObj : public Object {
  public:
    virtual ~CommunicatorObj() = default;
    virtual int getWorldSize() const = 0;
    virtual int getRank() const = 0;
    virtual int getLocalRank() const = 0;
};

} // namespace infini
