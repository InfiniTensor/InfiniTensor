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

#define CHECK_INFINI_ERROR(call)                                               \
    if (auto err = call; err != INFINI_STATUS_SUCCESS)                         \
    throw ::infini::Exception(                                                 \
        std::string("[") + __FILE__ + ":" + std::to_string(__LINE__) +         \
        "] operators error (" + #call + "): " + std::to_string(err))

} // namespace infini
