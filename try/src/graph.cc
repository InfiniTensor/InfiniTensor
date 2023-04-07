#include "graph.h"

size_t Tensor::size() const {
    return shape.empty() // fmt: new line
               ? 0
               : std::accumulate(shape.begin(), shape.end(), data_type.size(),
                                 [](auto acc, auto it) { return acc * it; });
}

size_t UniGraph::ID = 1;

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

void Candidates::push(UniGraph g, float score) {
    sorter.push({inner.size(), score});
    inner.push_back(std::move(g));
}

UniGraph Candidates::pop() {
    auto first = sorter.top().index;
    sorter.pop();
    return std::move(inner.at(first));
}
