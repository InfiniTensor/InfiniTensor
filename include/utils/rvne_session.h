#pragma once
#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace infini {
namespace rvne {
constexpr uint32_t RequestMagic = 0x52565131, ResponseMagic = 0x52565132;
constexpr uint32_t ProtocolVersion = 2, MaxPayloadBytes = 256U * 1024U * 1024U;
enum Command : uint32_t { LSMSequence = 1, MLP = 2 };
// Little endian, six uint32 words: magic, version, command, request id, bytes,
// status.
using Header = std::array<uint32_t, 6>;
inline void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class T> void read(std::istream &in, T *data, size_t count) {
    in.read(reinterpret_cast<char *>(data), count * sizeof(T));
    require(bool(in), "RVNE: truncated payload");
}
template <class T> void write(std::ostream &out, const T *data, size_t count) {
    out.write(reinterpret_cast<const char *>(data), count * sizeof(T));
    require(bool(out), "RVNE: payload write failed");
}
inline void end(std::istream &in) {
    require(in.peek() == std::char_traits<char>::eof(),
            "RVNE: trailing payload bytes");
}
} // namespace rvne
} // namespace infini

#include <memory>
#include <string>

namespace infini {
namespace rvne {
class Session;
// A lazy thread-local session per CPU graph run. Reference-only graphs spawn
// nothing. Nested kernel scopes reuse the graph scope; direct kernel calls get
// a short scope.
class SessionScope {
    bool ownsSessionScope;
    std::unique_ptr<Session> session;

  public:
    SessionScope();
    ~SessionScope();
    SessionScope(const SessionScope &) = delete;
    SessionScope &operator=(const SessionScope &) = delete;
    void finish();
    static std::string execute(Command command, const std::string &request,
                               size_t expectedBytes);
};
} // namespace rvne
} // namespace infini
