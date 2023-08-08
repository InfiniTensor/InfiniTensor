#include "bang/bang_gencode.h"

namespace infini {

BangGenCode::BangGenCode(std::vector<std::string> inputs_list, std::string output_value, std::vector<std::string> compute_list, int64_t total, int64_t worker) : inputs(inputs_list), output(output_value), compute(compute_list), totalNum(total), workerNum(worker) {}

std::string BangGenCode::genHead() {
  std::string temp = "#include <bang.h>\n";
  temp += "#define NRAM_SIZE " + std::to_string(nram_size) + "\n";
  return temp;
}

std::string BangGenCode::genNramSplit() {
  int64_t num = this->inputs.size();
  deal_size = nram_size / (num + 1);
  deal_size_align = ((deal_size) / (64) * (64));
  std::string temp = "";
  for(uint32_t i = 0; i < num; ++i) {
    temp += "__nram__ char nram_buffer" + std::to_string(i) + "[" + std::to_string(deal_size_align) + "];\n"; 
  }
  temp += "__nram__ char nram_buffer_output[" + std::to_string(deal_size_align) + "];\n"; 
  return temp;
}

std::string BangGenCode::genFunctionHead() {
  std::string temp = "";
  temp += "template<typename T>;\n";
  temp += "__mlu_device__ void FusionFunction(";
  temp += "T*" + this->output + ",\n";
  for(uint32_t i = 0; i < this->inputs.size(); ++i) {
    temp += "T*" + this->inputs[i] + ",\n";
  }
  temp += "int64_t taskNum) {\n";
  return temp;
}

std::string BangGenCode::genFunctionEnd() {
  return "}\n";
}

std::string BangGenCode::genTaskCycle() {
  int64_t num_per_core = this->totalNum / this->workerNum;
  int64_t num_rem = this->totalNum % this->workerNum;
  int64_t easy_job = num_per_core;
  int64_t hard_job = num_per_core + ( num_rem != 0 ? 1 : 0);
  std::string temp = "";
  temp += "int64_t my_job = taskId < " + std::to_string(num_rem) + " ? " + std::to_string(hard_job) + " : " + std::to_string(easy_job) + ";\n";
  temp += "int64_t start = ( taskId <= " + std::to_string(num_rem) + " ) ? ( taskId *" + std::to_string(hard_job) + " )\n";
  temp += ": ( " + std::to_string(num_rem) + " * " + std::to_string(hard_job) + " + ( taskId - " + std::to_string(num_rem) + ") * " + std::to_string(easy_job) + ");\n";
  return temp;
}

std::string BangGenCode::genOffset() {
  std::string temp = "";
  for(uint32_t i = 0; i < this->inputs.size(); ++i) {
    temp += "char *" + this->inputs[i] + "_start = (char*)" + this->inputs[i] + "start * sizeof(T);\n";
  }
  temp += "char *" + this->output + "_start = (char*)" + this->output + "start * sizeof(T);\n";
  return temp;
}

std::string BangGenCode::computeKernel(std::string size) {
  std::string temp = "";
  for(uint32_t i = 0; i < this->inputs.size(); ++i) {
    temp += "__memcpy(nram_buffer" + std::to_string(i) + ", " + this->inputs[i] + "_start, " + size + ", GDRAM2NRAM);\n";
  }
  ////////////////////////////////////////
  // Compute
  ////////////////////////////////////////
  temp += "__memcpy(" + this->output + "_start, nram_buffer_output, " + size + ", NRAM2GDRAM);\n"; 
  for(uint32_t i = 0; i < this->inputs.size(); ++i) {
    temp += this->inputs[i] + "_start += " + std::to_string(deal_size_align) + ";\n"; 
  }
  temp += this->output + "_start += " + std::to_string(deal_size_align) + ";\n";
  return temp;
}

std::string BangGenCode::genKernelCycle() {
  std::string temp = "";
  temp += "int64_t deal_num_align = " + std::to_string(deal_size_align) + " / sizeof(T);\n";
  temp += "int64_t repeat = my_job / deal_num_align;\n";
  temp += "int64_t rem = my_job % deal_num_align;\n";
  temp += "for(int i = 0; i < repeat; ++i) {;\n";
  temp += this->computeKernel(std::to_string(deal_size_align)); 
  temp += "}\n";
  temp += "if(rem){;\n";
  temp += this->computeKernel("rem * sizeof(T)"); 
  temp += "}\n";
  return temp;
}

std::string BangGenCode::genElementwiseFusion() {
  std::string temp = "";
  temp += genHead();
  temp += genNramSplit();
  temp += genFunctionHead();
  temp += genTaskCycle();
  temp += genOffset();
  temp += genKernelCycle();
  return temp;
}

}
