#include "test.h"
#include <string>
#include <iostream>
#include "tensor.pb.h"

namespace infini {

TEST(Protobuf, easy_test) {
    std::string test = "an unhappy tensor";
    tensor::tensor222 my_tensor;
    my_tensor.set_name("a happy tensor");
    my_tensor.SerializeToString(&test);
    std::cout << test << std::endl;
}

} // namespace infini
