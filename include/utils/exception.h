#pragma once
#include <stdexcept>
#include <string>

namespace infini {

class Exception : public std::runtime_error {
  public:
    Exception(const std::string &msg);
};

} // namespace infini
