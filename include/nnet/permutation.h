#pragma once
#include "common.h"
#include "expr.h"
#include <iostream>

namespace nnet {

class PermutationGenerator {
    vector<vector<Iterator>> from, to;
    vector<vector<size_t>> mapping;

  public:
    PermutationGenerator(vector<vector<Iterator>> _from,
                         vector<vector<Iterator>> _to);
    bool next();
    PtrMap<Iterator, Iterator> get() const;
};

template <typename T> class SubsetGenerator {
    vector<T> elements;
    int n, bitmap;

  public:
    SubsetGenerator(vector<T> elements, bool nonEmpty = 1)
        : elements(elements), n(elements.size()), bitmap((nonEmpty > 0)) {
        assert(n < 10);
    };
    bool next() { return ((++bitmap) < (1 << n) - 1); }
    vector<T> get() const {
        vector<T> ret;
        for (int i = 0; i < n; ++i)
            if (bitmap & (1 << i))
                ret.emplace_back(elements[i]);
        return ret;
    }
};

} // namespace nnet