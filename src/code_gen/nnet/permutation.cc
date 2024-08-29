#include "code_gen/nnet/permutation.h"
#include <algorithm>

namespace nnet {

PermutationGenerator::PermutationGenerator(vector<vector<Iterator>> _from,
                                           vector<vector<Iterator>> _to)
    : from(_from), to(_to), mapping(from.size()) {
    assert(from.size() == to.size());
    for (size_t i = 0; i < from.size(); ++i)
        for (size_t j = 0; j < from[i].size(); ++j)
            mapping[i].emplace_back(j);
}

bool PermutationGenerator::next() {
    if (mapping.empty())
        return false;
    for (int i = (int)mapping.size() - 1; i >= 0; --i) {
        if (std::next_permutation(mapping[i].begin(), mapping[i].end()))
            return true;
    }
    return false;
}

PtrMap<Iterator, Iterator> PermutationGenerator::get() const {
    if (mapping.empty())
        return {};
    PtrMap<Iterator, Iterator> ret;
    for (size_t i = 0; i < mapping.size(); ++i)
        for (size_t j = 0; j < mapping[i].size(); ++j)
            ret[from[i][j]] = to[i][mapping[i][j]];
    return ret;
}

} // namespace nnet