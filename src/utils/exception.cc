#include "utils/exception.h"

#ifdef BACKWARD_TRACE
#include "backward.hpp"

namespace backtrace = backward;

// signal handler
backtrace::SignalHandling sh;

namespace infini {
Exception::Exception(const std::string &msg) : std::runtime_error(msg) {
    backtrace::StackTrace st;
    st.load_here(32);
    backtrace::Printer p;
    p.print(st);
}
}; // namespace infini

#else

namespace infini {
Exception::Exception(const std::string &msg) : std::runtime_error(msg) {}
} // namespace infini

#endif
