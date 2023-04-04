#include "../src/graph.h"
#include <iostream>

int main() {
    try {
        Graph<TensorInfo> g;
        auto input = g.push_operator(OpType::Input, {},
                                     {{{{1, 2, 3}, {DataTypeId::FLOAT}}}});
        g.push_data(input, Data::cpu<float>({1, 2, 3, 4, 5, 6, 7}));
        g.inputs();
        return 0;
    } catch (const char *e) {
        std::cerr << "[ERROR] " << e << std::endl;
        return 1;
    }
}
