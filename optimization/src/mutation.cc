#include "mutation.h"
#include <unordered_set>

Vec<std::pair<Unigraph, SingleOperator>> split_each(Unigraph &&g) {
    Vec<std::pair<Unigraph, SingleOperator>> ans;
    for (auto &op : g.operators) {
        auto &[g, t] = ans.emplace_back();
        g.push_operator(op.op_type, op.inputs, op.outputs);
    }
    return ans;
}

float memory_usage(Unigraph const &g) {
    std::unordered_set<size_t> mark;
    uintptr_t memory;
    for (const auto &op : g.operators)
        for (const auto &t : op.outputs)
            if (mark.insert(reinterpret_cast<uintptr_t>(t.get())).second)
                memory += t->size();
    return 1e6f / static_cast<float>(memory);
}
