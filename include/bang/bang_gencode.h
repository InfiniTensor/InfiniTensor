#pragma once
#include "cnrt.h"
#include <string>
#include <vector>

namespace infini {

class BangGenCode {
  private:
    int64_t nram_size = 589824;
    int64_t deal_size;
    int64_t deal_size_align;
    std::vector<std::string> inputs;
    std::string output;
    std::vector<std::string> compute;
    int64_t totalNum;
    int64_t workerNum;
  public:
    BangGenCode(std::vector<std::string> inputs_list, std::string output_value, std::vector<std::string> compute_list, int64_t total, int64_t worker);
  private:
    std::string genHead();
    std::string genNramSplit(int64_t num);
    std::string genFunctionHead(std::vector<std::string> inputs, std::string output); 
    std::string genFunctionEnd();
    std::string genTaskCycle(int64_t taskNum, int64_t workerNum);
    std::string genOffset(std::vector<std::string> inputs, std::string output);
    std::string computeKernel(std::vector<std::string> inputs, std::string output, std::vector<std::string> compute, std::string size);
    std::string genKernelCycle();
  public:
    std::string genElementwiseFusion();
};
}
