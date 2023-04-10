#include "graph.h"
#include <unordered_set>

static size_t GRAPH_ID = 1;

Unigraph::Unigraph() : id(GRAPH_ID++) {}

Unigraph::Unigraph(Unigraph &&others)
    : id(std::exchange(others.id, 0)), operators(std::move(others.operators)) {}

Unigraph::~Unigraph() {
    for (auto &op : operators) {
        for (auto &i : op.inputs)
            i->target.erase(i->target.find(this->id));
        for (auto &o : op.outputs)
            o->source.erase(o->source.find(this->id));
    }
}

Unigraph &Unigraph::operator=(Unigraph &&others) {
    if (this == &others)
        return *this;

    for (auto &op : operators) {
        for (auto &i : op.inputs)
            i->target.erase(i->target.find(this->id));
        for (auto &o : op.outputs)
            o->source.erase(o->source.find(this->id));
    }

    this->id = std::exchange(others.id, 0);
    this->operators = std::move(others.operators);

    return *this;
}

OpRef Unigraph::push_operator( // fmt: new line
    OpType op_type,            //
    Vec<Arc<Tensor>> inputs,   //
    Vec<Arc<Tensor>> outputs   //
) {
    auto ans = OpRef{this->id, operators.size()};

    size_t i = 0;
    for (auto &input : inputs) {
        auto it = input->target.find(ans.graph);
        if (it == input->target.end())
            input->target[ans.graph] = {{ans.op, i++}};
        else
            it->second.push_back({ans.op, i++});
    }
    i = 0;
    for (auto &output : outputs) {
        auto it = output->source.find(ans.graph);
        if (it == output->source.end())
            output->source[ans.graph] = {ans.op, i++};
        else
            throw "tensor source exist";
    }

    this->operators.push_back({
        op_type,            // fmt: new line
        std::move(inputs),  //
        std::move(outputs), //
    });
    return ans;
}

bool Mutant::operator<(Mutant const &others) const {
    return this->score < others.score;
}

bool Mutant::operator>(Mutant const &others) const {
    return this->score > others.score;
}

Mutant::Mutant(Unigraph &&g) : graph(std::move(g)) {}
Mutant::Mutant(Mutant &&others) : graph(std::move(others.graph)) {}
Mutant &Mutant::operator=(Mutant &&others) {
    if (this != &others)
        this->graph = std::move(others.graph);
    return *this;
}

Partition::Partition(Unigraph &&g, Func const &f) {
    auto graph = f(std::move(g));
    for (auto &sub : graph)
        this->graph.emplace_back().emplace_back(std::move(sub));
}

Mutation::Mutation(Partition &&p, Func const &f) : graph(std::move(p.graph)) {
    for (auto &sub : graph)
        for (auto &m : f(sub.front().graph))
            sub.emplace_back(std::move(m));
}

Rating::Rating(Mutation &&m, Func const &f) : graph(std::move(m.graph)) {
    for (auto &sub : graph) {
        for (auto &c : sub)
            c.score = f(c.graph);
        std::sort(sub.begin(), sub.end());
    }
}

Vec<Unigraph> split_each(Unigraph &&g) {
    Vec<Unigraph> ans;
    for (auto &op : g.operators)
        ans.emplace_back().push_operator(op.op_type, op.inputs, op.outputs);
    return ans;
}

float memory_usage(Unigraph const &g) {
    std::unordered_set<size_t> mark;
    uintptr_t memory;
    for (const auto &op : g.operators)
        for (const auto &t : op.outputs)
            if (mark.insert(reinterpret_cast<uintptr_t>(t.get())).second)
                memory += t->size();
    return 1.0f / static_cast<float>(memory);
}
