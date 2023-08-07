#pragma once
#include "cnrt.h"
#include <string>

namespace infini {

class BangGenCode {
  private:
    int64_t nram_size = 589824;
  public:
    std::string genHead() {
      std::string temp = "#include <bang.h>\n";
      temp += "#define NRAM_SIZE " + std::to_string(nram_size) + "\n";
      return temp;
    }
    std::string genNramSplit(int64_t num) {
      int64_t deal_size = nram_size / (num + 1);
      int64_t deal_size_align = ((deal_size) / (64) * (64));
      std::string temp = "";
      for(int64_t i = 0; i < num; ++i) {
        temp += "__nram__ char nram_buffer" + std::to_string(i) + "[" + std::to_string(deal_size_align) + "]\n"; 
      }
      temp += "__nram__ char nram_buffer_output[" + std::to_string(deal_size_align) + "]\n"; 
    }
    return temp;
    std::string genElementwiseFusion() {
    }
}
}
