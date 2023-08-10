#pragma once
#include "object.h"
#include "ref.h"

namespace infini {

// base class
class CommunicatorObj : public Object {
  public:
    virtual ~CommunicatorObj() = default;
};

} // namespace infini
