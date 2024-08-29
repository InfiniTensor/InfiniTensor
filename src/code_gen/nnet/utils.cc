#include "code_gen/nnet/common.h"
namespace nnet {

std::string pointer_to_hex(void *i) {
    std::stringstream stream;
    // stream << "0x" << std::setfill('0') << std::setw(sizeof(void *) * 2) <<
    // std::hex
    //        << i;
    stream << std::hex << i;
    return stream.str();
}
} // namespace nnet