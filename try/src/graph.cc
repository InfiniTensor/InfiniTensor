#include "graph.h"
#include <numeric>
#include <unordered_set>

size_t Tensor::size() const {
    return shape.empty() // fmt: new line
               ? 0
               : std::accumulate(shape.begin(), shape.end(), data_type.size(),
                                 [](auto acc, auto it) { return acc * it; });
}

static size_t GRAPH_ID = 1;

UniGraph::UniGraph() : id(GRAPH_ID++) {}

UniGraph::UniGraph(UniGraph &&others)
    : id(std::exchange(others.id, 0)), operators(std::move(others.operators)) {}

UniGraph::~UniGraph() {
    for (auto &op : operators) {
        for (auto &i : op.inputs)
            i->target.erase(i->target.find(this->id));
        for (auto &o : op.outputs)
            o->source.erase(o->source.find(this->id));
    }
}

UniGraph &UniGraph::operator=(UniGraph &&others) {
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

OpRef UniGraph::push_operator( // fmt: new line
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

bool Candidate::operator<(Candidate const &others) const {
    return this->score < others.score;
}

bool Candidate::operator>(Candidate const &others) const {
    return this->score > others.score;
}

Candidate::Candidate(UniGraph &&g) : graph(std::move(g)) {}
Candidate::Candidate(Candidate &&others) : graph(std::move(others.graph)) {}
Candidate &Candidate::operator=(Candidate &&others) {
    if (this != &others)
        this->graph = std::move(others.graph);
    return *this;
}

Partition::Partition(UniGraph &&g, Func const &f) {
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

Vec<UniGraph> split_each(UniGraph &&g) {
    Vec<UniGraph> ans;
    for (auto &op : g.operators)
        ans.emplace_back().push_operator(op.op_type, op.inputs, op.outputs);
    return ans;
}

float memory_usage(UniGraph const &g) {
    std::unordered_set<size_t> mark;
    uintptr_t memory;
    for (const auto &op : g.operators)
        for (const auto &t : op.outputs)
            if (mark.insert(reinterpret_cast<uintptr_t>(t.get())).second)
                memory += t->size();
    return static_cast<float>(memory);
}
