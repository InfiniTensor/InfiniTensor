#include "../include/optimization/common.h"
#include <iostream>
#include <unordered_set>

using namespace optimization;

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

        auto p = Partition<pass::SingleOperator>(std::move(g), pass::partition);
        auto m = Mutation<pass::SingleOperator>(
            std::move(p),
            [](const auto &g, const auto &t) { return Vec<Unigraph>{}; });
        auto r = Rating<pass::SingleOperator>(std::move(m), memory_usage);
        auto ans = r.build(Vec<size_t>(r.size().size(), 0));

        return 0;
    } catch (const char *e) {
        std::cerr << "[ERROR] " << e << std::endl;
        return 1;
    }
}
