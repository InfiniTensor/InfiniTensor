#pragma once
#include <stdexcept>
#include <string>

namespace infini {

class Exception : public std::runtime_error {
  protected:
    std::string info;

  public:
    Exception(const std::string &msg);

    Exception &operator<<(const std::string &str) {
        info += str;
        return *this;
    }

    const char *what() const noexcept override { return info.c_str(); }
};

} // namespace infini
