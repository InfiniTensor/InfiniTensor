#include "bang/bang_gencode.h"

#include "test.h"

namespace infini {

void testCodegen() {
  std::vector<std::string> inputs;
  std::vector<std::string> compute;
  inputs.push_back("inputa");
  inputs.push_back("inputb");
  compute.push_back("add");
  std::string output;
  BangGenCode gen(inputs, output, compute, 1024, 12);
  std::cout << gen.genElementwiseFusion();

  EXPECT_TRUE(1);
}

TEST(cnnl_Clip, run) {
    testCodegen();
}

} // namespace infini
