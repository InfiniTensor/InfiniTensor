#include "../src/graph.h"
#include <iostream>

int main() {
    try {
        Graph<TensorInfo> g;
        auto a = g.push_operator(                           // fmt: new line
            OpType::Input,                                  // op_type
            {},                                             // inputs
            {Outlet(TensorInfo{{1, 1, 2, 3}, ty<float>()})} // outputs
        );
        g.push_data(a, Data::cpu<float>({1, 2, 3, 4, 5, 6}));

        auto b = g.push_operator(                           // fmt: new line
            OpType::Input,                                  // op_type
            {},                                             // inputs
            {Outlet(TensorInfo{{1, 1, 3, 1}, ty<float>()})} // outputs
        );
        g.push_data(b, Data::cpu<float>({1, 2, 3}));

        auto matmul = g.push_operator(                      // fmt: new line
            OpType::MatMul,                                 // op_type
            {a[0], b[0]},                                   // inputs
            {Outlet(TensorInfo{{1, 1, 2, 1}, ty<float>()})} // outputs
        );

        g.push_operator(    // fmt: new line
            OpType::Output, // op_type
            {matmul[0]},    // inputs
            {}              // outputs
        );

        std::cout << "inputs: ";
        for (auto it : g.inputs()) {
            std::cout << it << " ";
        }
        std::cout << std::endl;

        std::cout << "outputs: ";
        for (auto it : g.outputs()) {
            std::cout << it << " ";
        }
        std::cout << std::endl;

        return 0;
    } catch (const char *e) {
        std::cerr << "[ERROR] " << e << std::endl;
        return 1;
    }
}
