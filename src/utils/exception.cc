#include "utils/exception.h"

#ifdef BACKWARD_TRACE
#include "backward.hpp"

namespace my_backtrace = backward;

// signal handler
my_backtrace::SignalHandling sh;

namespace infini {
Exception::Exception(const std::string &msg) : std::runtime_error(msg) {
    my_backtrace::StackTrace st;
    st.load_here(32);
    my_backtrace::Printer p;
    p.print(st);
}
}; // namespace infini

#else

namespace infini {
Exception::Exception(const std::string &msg) : std::runtime_error(msg) {}
} // namespace infini

#endif
