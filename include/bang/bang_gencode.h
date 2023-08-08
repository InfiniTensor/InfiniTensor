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
    BangGenCode(std::vector<std::string> inputs_list, std::string output_value, std::vector<std::string> compute_list, int64_t total, int64_t worker):
               inputs(inputs_list), output(output_value), compute(compute_list), totalNum(total), workerNum(worker) {
    }
  private:
    std::string genHead() {
      std::string temp = "#include <bang.h>\n";
      temp += "#define NRAM_SIZE " + std::to_string(nram_size) + "\n";
      return temp;
    }
    std::string genNramSplit(int64_t num) {
      deal_size = nram_size / (num + 1);
      deal_size_align = ((deal_size) / (64) * (64));
      std::string temp = "";
      for(uint32_t i = 0; i < num; ++i) {
        temp += "__nram__ char nram_buffer" + std::to_string(i) + "[" + std::to_string(deal_size_align) + "];\n"; 
      }
      temp += "__nram__ char nram_buffer_output[" + std::to_string(deal_size_align) + "];\n"; 
      return temp;
    }
    std::string genFunctionHead(std::vector<std::string> inputs, std::string output) {
      std::string temp = "";
      temp += "template<typename T>;\n";
      temp += "__mlu_device__ void FusionFunction(";
      temp += "T*" + output + ",\n";
      for(uint32_t i = 0; i < inputs.size(); ++i) {
        temp += "T*" + inputs[i] + ",\n";
      }
      temp += "int64_t taskNum) {\n";
      return temp;
    }
    std::string genFunctionEnd() {
      return "}\n";
    }
    std::string genTaskCycle(int64_t taskNum, int64_t workerNum) {
      int64_t num_per_core = taskNum / workerNum;
      int64_t num_rem = taskNum % workerNum;
      int64_t easy_job = num_per_core;
      int64_t hard_job = num_per_core + ( num_rem != 0 ? 1 : 0);
      std::string temp = "";
      temp += "int64_t my_job = taskId < " + std::to_string(num_rem) + " ? " + std::to_string(hard_job) + " : " + std::to_string(easy_job) + ";\n";
      temp += "int64_t start = ( taskId <= " + std::to_string(num_rem) + " ) ? ( taskId *" + std::to_string(hard_job) + " )\n";
      temp += ": ( " + std::to_string(num_rem) + " * " + std::to_string(hard_job) + " + ( taskId - " + std::to_string(num_rem) + ") * " + std::to_string(easy_job) + ");\n";
      return temp;
    }
    std::string genOffset(std::vector<std::string> inputs, std::string output) {
      std::string temp = "";
      for(uint32_t i = 0; i < inputs.size(); ++i) {
        temp += "char *" + inputs[i] + "_start = (char*)" + inputs[i] + "start * sizeof(T);\n";
      }
      temp += "char *" + output + "_start = (char*)" + output + "start * sizeof(T);\n";
      return temp;
    }
    std::string computeKernel(std::vector<std::string> inputs, std::string output, std::vector<std::string> compute, std::string size) {
      std::string temp = "";
      for(uint32_t i = 0; i < inputs.size(); ++i) {
        temp += "__memcpy(nram_buffer" + std::to_string(i) + ", " + inputs[i] + "_start, " + size + ", GDRAM2NRAM);\n";
      }
      ////////////////////////////////////////
      // Compute
      ////////////////////////////////////////
      temp += "__memcpy(" + output + "_start, nram_buffer_output, " + size + ", NRAM2GDRAM);\n"; 
      for(uint32_t i = 0; i < inputs.size(); ++i) {
        temp += inputs[i] + "_start += " + std::to_string(deal_size_align) + ";\n"; 
      }
      temp += output + "_start += " + std::to_string(deal_size_align) + ";\n";
      return temp;
    }
    std::string genKernelCycle() {
      std::string temp = "";
      temp += "int64_t deal_num_align = " + std::to_string(deal_size_align) + " / sizeof(T);\n";
      temp += "int64_t repeat = my_job / deal_num_align;\n";
      temp += "int64_t rem = my_job % deal_num_align;\n";
      temp += "for(int i = 0; i < repeat; ++i) {;\n";
      temp += this->computeKernel(inputs, output, compute, std::to_string(deal_size_align)); 
      temp += "}\n";
      temp += "if(rem){;\n";
      temp += this->computeKernel(inputs, output, compute, "rem * sizeof(T)"); 
      temp += "}\n";
      return temp;
    }
  public:
    std::string genElementwiseFusion() {
      std::string temp = "";
      temp += genHead();
      temp += genNramSplit(inputs.size());
      temp += genFunctionHead(inputs, output);
      temp += genTaskCycle(this->totalNum, this->workerNum);
      temp += genOffset(inputs, output);
      temp += genKernelCycle();
      return temp;
    }
};
}
