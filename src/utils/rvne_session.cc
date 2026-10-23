#include "utils/rvne_session.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>
#ifdef __linux__
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace infini {
namespace rvne {
using Clock = std::chrono::steady_clock;
static thread_local SessionScope *active = nullptr;
static std::string env(const char *key, const std::string &fallback = "") {
    const char *value = std::getenv(key);
    return value ? value : fallback;
}
#ifdef __linux__
class Session {
    int fd = -1;
    pid_t pid = -1;
    uint32_t sequence = 0;
    long timeout = 120000;
    bool closed = false, keep = false;
    std::string directory, log, runner, status = "ok";
    Clock::time_point started = Clock::now();
    nlohmann::json requests = nlohmann::json::array();

    [[noreturn]] void fail(const std::string &message,
                           const char *result = "failed") {
        status = result;
        throw std::runtime_error("RVNE: " + message);
    }
    void transfer(void *data, size_t bytes, bool writing,
                  Clock::time_point deadline) {
        auto *cursor = static_cast<char *>(data);
        while (bytes) {
            auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - Clock::now())
                            .count();
            if (left <= 0)
                fail("QEMU timeout", "timeout");
            pollfd descriptor{fd, short(writing ? POLLOUT : POLLIN), 0};
            int ready =
                poll(&descriptor, 1, int(std::min<int64_t>(left, 1000)));
            if (ready < 0) {
                if (errno == EINTR)
                    continue;
                fail("channel poll failed");
            }
            if (!ready)
                continue;
            ssize_t count =
                writing ? send(fd, cursor, bytes, MSG_NOSIGNAL | MSG_DONTWAIT)
                        : recv(fd, cursor, bytes, MSG_DONTWAIT);
            if (count < 0 &&
                (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                continue;
            if (count <= 0)
                fail("runner failed or channel closed");
            cursor += count;
            bytes -= size_t(count);
        }
    }
    void cleanup(bool terminate) noexcept {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (pid > 0) {
            if (terminate)
                kill(pid, SIGKILL);
            int result = 0;
            while (waitpid(pid, &result, 0) < 0 && errno == EINTR) {
            }
            pid = -1;
        }
    }
    void audit() noexcept {
        try {
            auto path = env("INFINI_RVNE_AUDIT");
            if (path.empty())
                return;
            nlohmann::json entry{
                {"engine", "qemu"},
                {"status", status},
                {"pid", auditPid},
                {"mode", "persistent"},
                {"requests", requests},
                {"runner", runner},
                {"milliseconds", std::chrono::duration<double, std::milli>(
                                     Clock::now() - started)
                                     .count()},
                {"io_directory", directory},
                {"io_retained", keep}};
            auto line = entry.dump() + "\n";
            int out = open(path.c_str(),
                           O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
            if (out >= 0) {
                auto ignored = ::write(out, line.data(), line.size());
                (void)ignored;
                ::close(out);
            }
        } catch (...) {
        }
    }
    pid_t auditPid = -1;

  public:
    Session() {
        auto qemu = env("INFINI_RVNE_QEMU"),
             sysroot = env("INFINI_RVNE_SYSROOT");
        runner = env("INFINI_RVNE_RUNNER");
        require(!qemu.empty() && !sysroot.empty() && !runner.empty(),
                "RVNE: set INFINI_RVNE_QEMU, INFINI_RVNE_SYSROOT and "
                "INFINI_RVNE_RUNNER");
        require(access(qemu.c_str(), X_OK) == 0 &&
                    access(runner.c_str(), X_OK) == 0,
                "RVNE: QEMU or runner is not executable");
        try {
            timeout = std::stol(env("INFINI_RVNE_TIMEOUT_MS", "120000"));
        } catch (...) {
        }
        timeout = std::max(1L, std::min(600000L, timeout));
        keep = env("INFINI_RVNE_KEEP_IO") == "1";
        auto pattern = env("TMPDIR", "/tmp") + "/infini-rvne-session-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back(0);
        char *created = mkdtemp(name.data());
        require(created, "RVNE: cannot create temporary directory");
        directory = created;
        log = directory + "/runner.log";
        int channel[2] = {-1, -1};
        posix_spawn_file_actions_t actions;
        bool initialized = false;
        try {
            require(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0,
                               channel) == 0,
                    "RVNE: socketpair failed");
            require(posix_spawn_file_actions_init(&actions) == 0,
                    "RVNE: spawn actions failed");
            initialized = true;
            int code = posix_spawn_file_actions_adddup2(&actions, channel[1],
                                                        STDIN_FILENO);
            if (!code)
                code = posix_spawn_file_actions_adddup2(&actions, channel[1],
                                                        STDOUT_FILENO);
            if (!code)
                code = posix_spawn_file_actions_addclose(&actions, channel[0]);
            if (!code)
                code = posix_spawn_file_actions_addclose(&actions, channel[1]);
            if (!code)
                code = posix_spawn_file_actions_addopen(
                    &actions, STDERR_FILENO, log.c_str(),
                    O_WRONLY | O_CREAT | O_TRUNC, 0600);
            require(code == 0, "RVNE: spawn redirection failed");
            std::vector<std::string> args{qemu, "-L", sysroot, runner,
                                          "--serve"};
            std::vector<char *> argv;
            for (auto &arg : args)
                argv.push_back(const_cast<char *>(arg.c_str()));
            argv.push_back(nullptr);
            int result = posix_spawn(&pid, qemu.c_str(), &actions, nullptr,
                                     argv.data(), environ);
            require(result == 0, "RVNE: cannot launch QEMU");
            auditPid = pid;
            fd = channel[0];
            channel[0] = -1;
            ::close(channel[1]);
            channel[1] = -1;
            posix_spawn_file_actions_destroy(&actions);
        } catch (...) {
            if (initialized)
                posix_spawn_file_actions_destroy(&actions);
            for (int value : channel)
                if (value >= 0)
                    ::close(value);
            cleanup(true);
            unlink(log.c_str());
            rmdir(directory.c_str());
            throw;
        }
    }
    ~Session() {
        try {
            close();
        } catch (...) {
        }
        if (!keep) {
            unlink(log.c_str());
            rmdir(directory.c_str());
        }
    }
    std::string execute(Command command, const std::string &request,
                        size_t expected) {
        require(!closed && status == "ok", "RVNE: session is no longer usable");
        require(request.size() <= MaxPayload && expected <= MaxPayload,
                "RVNE: payload too large");
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        Header header{RequestMagic,
                      Version,
                      command,
                      ++sequence,
                      uint32_t(request.size()),
                      0};
        requests.push_back(
            {{"id", sequence},
             {"command", command == LSM ? "LSMSequence" : "MLP"}});
        try {
            transfer(header.data(), sizeof(header), true, deadline);
            transfer(const_cast<char *>(request.data()), request.size(), true,
                     deadline);
            Header response{};
            transfer(response.data(), sizeof(response), false, deadline);
            if (response[0] != ResponseMagic || response[1] != Version ||
                response[2] != command || response[3] != sequence ||
                response[4] > MaxPayload || response[5] > 1)
                fail("response protocol mismatch");
            if ((!response[5] && response[4] != expected) ||
                (response[5] && response[4] > 8192))
                fail("response shape/length mismatch");
            std::string payload(response[4], '\0');
            transfer(payload.data(), payload.size(), false, deadline);
            if (response[5])
                fail("runner failed: " + payload);
            requests.back()["status"] = "ok";
            return payload;
        } catch (...) {
            if (status == "ok")
                status = "failed";
            requests.back()["status"] = status;
            cleanup(true);
            throw;
        }
    }
    void close() {
        if (closed)
            return;
        closed = true;
        if (status == "ok" && pid > 0) {
            ::shutdown(fd, SHUT_WR);
            ::close(fd);
            fd = -1;
            auto deadline = Clock::now() +
                            std::chrono::milliseconds(std::min(1000L, timeout));
            int result = 0;
            while (true) {
                pid_t done = waitpid(pid, &result, WNOHANG);
                if (done == pid) {
                    pid = -1;
                    if (!WIFEXITED(result) || WEXITSTATUS(result))
                        status = "failed";
                    break;
                }
                if (done < 0 && errno != EINTR) {
                    status = "failed";
                    break;
                }
                if (Clock::now() >= deadline) {
                    status = "timeout";
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        cleanup(true);
        audit();
        if (status != "ok")
            throw std::runtime_error(
                "RVNE: runner failed or timeout closing session");
    }
};
#else
class Session {
  public:
    Session() { throw std::runtime_error("RVNE: QEMU session requires Linux"); }
    void close() {}
    std::string execute(Command, const std::string &, size_t) { return {}; }
};
#endif
SessionScope::SessionScope() : owner(active == nullptr) {
    if (owner)
        active = this;
}
SessionScope::~SessionScope() {
    if (owner) {
        session.reset();
        active = nullptr;
    }
}
void SessionScope::finish() {
    if (owner && session)
        session->close();
}
std::string SessionScope::execute(Command command, const std::string &request,
                                  size_t expectedBytes) {
    require(active != nullptr, "RVNE: missing execution scope");
    if (!active->session)
        active->session = std::make_unique<Session>();
    return active->session->execute(command, request, expectedBytes);
}
} // namespace rvne
} // namespace infini
