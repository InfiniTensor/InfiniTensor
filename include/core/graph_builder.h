#pragma once
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "graph.h"
#include <cstddef>
#include <unordered_map>
#include <variant>
#include <vector>

namespace infini {

// struct Shape {
//     std::vector<int64_t> dims;
// };

class GraphBuilder {
    using Int = long long;
    using Ints = std::vector<long long>;
    using Float = double;
    using Floats = std::vector<double>;
    using String = std::string;
    using Strings = std::vector<std::string>;
    using Attribute = std::variant<Int, Ints, Float, Floats, String, Strings>;
    using Attributes = std::unordered_map<std::string, Attribute>;

    struct OpInfo {
        std::string opType;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        Attributes attributes;

        std::unordered_set<std::string> unknownInputs;
    };

    Graph graph;
    std::unordered_map<std::string, Tensor> tensors;
    std::vector<OpInfo> unknownOps;

    template <class T>
    std::optional<T> takeAttribute(Attributes &attributes, const char *name) {
        auto it = attributes.find(name);
        if (it != attributes.end()) {
            auto value = std::get<T>(std::move(it->second));
            attributes.erase(it);
            return {value};
        }
        return std::nullopt;
    }

  public:
    void addTensor(std::string name, //
                   Shape shape,      //
                   DataType dataType);

    void addOperation(std::string opType,               //
                      std::vector<std::string> inputs,  //
                      std::vector<std::string> outputs, //
                      Attributes attributes);
};

} // namespace infini

#endif // GRAPH_BUILDER_H
