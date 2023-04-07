#include "../src/graph.h"
#include <iostream>
#include <unordered_set>

int main() {
    try {
        Unigraph g;
        auto a = Tensor::share( // fmt: new line
            {1, 1, 2, 3},       //
            ty<float>(),        //
            Data::cpu<float>({1, 2, 3, 4, 5, 6}));

        auto b = Tensor::share( // fmt: new line
            {1, 1, 3, 1},       //
            ty<float>(),        //
            Data::cpu<float>({1, 2, 3}));

        auto c = Tensor::share( // fmt: new line
            {1, 1, 2, 1},       //
            ty<float>(),        //
            {});

        auto matmul = g.push_operator( // fmt: new line
            OpType::MatMul,            // op_type
            {a, b},                    // inputs
            {c}                        // outputs
        );

        auto p = Partition(std::move(g), split_each);
        auto m = Mutation(std::move(p),
                          [](const auto &g) { return Vec<Unigraph>{}; });
        auto r = Rating(std::move(m), memory_usage);

        return 0;
    } catch (const char *e) {
        std::cerr << "[ERROR] " << e << std::endl;
        return 1;
    }
}
