#pragma once

#include <string>
#include <vector>

namespace memb {

class CodeBuffer {
  public:
    std::vector<std::string> buf;
    CodeBuffer() { buf.clear(); }
    void emit(std::string inst) { buf.emplace_back(inst); }
    std::string toString() {
        std::string tmp = "";
        for (int i = 0; i < int(buf.size()); i++) {
            tmp += buf[i] + "\n";
        }
        return tmp;
    }
};

} // namespace memb
