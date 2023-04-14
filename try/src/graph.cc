#include "graph.h"

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
