#include "utils/exception.h"

#ifdef BACKWARD_TRACE
#include "backward.hpp"

namespace host_backtrace = backward;

// signal handler
host_backtrace::SignalHandling sh;

namespace infini {
Exception::Exception(const std::string &msg)
    : std::runtime_error(msg), info(msg) {
    backward_trace::StackTrace st;
    st.load_here(32);
    host_backtrace::Printer p;
    p.print(st);
}
}; // namespace infini

#else

namespace infini {
Exception::Exception(const std::string &msg) : std::runtime_error(msg) {}
} // namespace infini

#endif
