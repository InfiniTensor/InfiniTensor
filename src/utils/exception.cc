#include "utils/exception.h"

#ifdef BACKWARD_TRACE
#include "backward.hpp"

namespace backward_trace = backward;

// signal handler
backward_trace::SignalHandling sh;

namespace infini {
Exception::Exception(const std::string &msg)
    : std::runtime_error(msg), info(msg) {
    backward_trace::StackTrace st;
    st.load_here(32);
    backward_trace::Printer p;
    p.print(st);
}
}; // namespace infini

#else

namespace infini {
// `info` backs the overridden `what()`, so it must carry the message here too,
// otherwise every assertion message is lost when tracing is disabled.
Exception::Exception(const std::string &msg)
    : std::runtime_error(msg), info(msg) {}
} // namespace infini

#endif
