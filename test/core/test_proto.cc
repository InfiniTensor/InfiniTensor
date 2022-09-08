#include "data.pb.h"
#include "test.h"
#include <iostream>
#include <string>
#include <vector>

namespace infini {

TEST(Protobuf, easy_test) {
    std::string test = "an unhappy tensor";
    std::vector<int> a{1, 2, 2, 3};
    std::vector<float> data{1.2, 2.3, 3.4, 4.5,  5.6,  6.7,
                            7.8, 8.9, 9.1, 10.1, 11.2, 12.3};
    data::Tensor my_tensor;
    my_tensor.set_id("a happy tensor");
    my_tensor.mutable_shape()->CopyFrom({a.begin(), a.end()});
    my_tensor.set_layout(data::LAYOUT_NHWC);
    my_tensor.set_dtype(data::DTYPE_FLOAT);
    my_tensor.mutable_data_float()->CopyFrom({data.begin(), data.end()});
    my_tensor.SerializeToString(&test);
    std::cout << test << std::endl;
}

} // namespace infini
