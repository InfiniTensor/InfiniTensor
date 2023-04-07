#include "../src/graph.h"
#include "../src/partitions.h"
#include <iostream>
#include <unordered_set>

int main() {
    try {
        UniGraph g;
        auto a = Arc<Tensor>(new Tensor{
            {1, 1, 2, 3},                        // fmt: new line
            ty<float>(),                         //
            Data::cpu<float>({1, 2, 3, 4, 5, 6}) //
        });

        auto b = Arc<Tensor>(new Tensor{
            {1, 1, 3, 1},               // fmt: new line
            ty<float>(),                //
            Data::cpu<float>({1, 2, 3}) //
        });

        auto c = Arc<Tensor>(new Tensor{
            {1, 1, 2, 1}, // fmt: new line
            ty<float>(),  //
        });

        auto matmul = g.push_operator( // fmt: new line
            OpType::MatMul,            // op_type
            {a, b},                    // inputs
            {c}                        // outputs
        );

        auto graph = split_each(std::move(g), [](UniGraph const &g) {
            std::unordered_set<size_t> mark;
            uintptr_t memory;
            for (const auto &op : g.operators) {
                for (const auto &t : op.inputs)
                    if (mark.insert(reinterpret_cast<uintptr_t>(t.get()))
                            .second) {
                        memory += t->size();
                    }
                for (const auto &t : op.outputs)
                    if (mark.insert(reinterpret_cast<uintptr_t>(t.get()))
                            .second) {
                        memory += t->size();
                    }
            }
            return static_cast<float>(memory);
        });

        return 0;
    } catch (const char *e) {
        std::cerr << "[ERROR] " << e << std::endl;
        return 1;
    }
}
