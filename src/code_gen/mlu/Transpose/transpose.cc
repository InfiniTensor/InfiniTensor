#include "code_gen/transpose.h"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

namespace tpm {

bool isVariable(char c) { return (c >= 'a' && c <= 'z'); }

int getVarIndex(char c) {
    assert(isVariable(c));
    return c - 'a';
}

char getVarName(int ind) {
    assert(ind >= 0 && ind < 26);
    return 'a' + ind;
}

tpm::DecisionTree *tpm::TransSplit::putTranspose(tpm::DecisionTree *tree,
                                                 std::vector<int> &dimsize) {
    assert(index >= 0 && index < (int)dimsize.size() && in_dim > 0);
    assert(extr == 0 || copys == 0);
    char _newvar[100];
    if (copys == 0) {
        sprintf(_newvar, "(%c*%d+%c)", 'a' + index, in_dim + extr,
                'a' + index + 1);
        std::string newvar = _newvar;
        dfsSplit(tree, newvar);
        dimsize.insert(dimsize.begin() + index + 1, in_dim);
        dimsize[index] = (dimsize[index] - 1) / (in_dim + extr) + 1;
    } else {
        assert(dimsize[index] == in_dim * 2);
        sprintf(_newvar, "(%c*%d+%c-%c*%d)", 'a' + index, in_dim,
                'a' + index + 1, 'a' + index, copys);
        std::string newvar = _newvar;
        dfsSplit(tree, newvar);
        dimsize.insert(dimsize.begin() + index + 1, in_dim + copys);
        dimsize[index] = dimsize[index] / in_dim;
    }
    return tree;
}

void tpm::TransSplit::dfsSplit(DecisionTree *tree, std::string newvar) {
    if (tree == nullptr)
        return;
    std::vector<std::string> mapLambda = tree->getCond();
    for (auto &exp : mapLambda) {
        std::string newexp = "";
        for (auto c : exp) {
            if (isVariable(c) && getVarIndex(c) > index)
                newexp += (c + 1);
            else if (isVariable(c) && getVarIndex(c) == index)
                newexp += newvar;
            else
                newexp += c;
        }
        exp = newexp;
    }
    tree->putCond(mapLambda);
    dfsSplit(tree->getThenNode(), newvar);
    dfsSplit(tree->getElseNode(), newvar);
}

void tpm::TransSplit::print() {
    std::cout << "TransSplit(index=" << index << ", split_factor=" << in_dim
              << ")" << std::endl;
}

tpm::DecisionTree *tpm::TransFuse::putTranspose(tpm::DecisionTree *tree,
                                                std::vector<int> &dimsize) {
    assert(index >= 0 && index + 1 < (int)dimsize.size());
    int in_dim = dimsize[index + 1];
    char _newvar[100];
    if (dels > 0) {
        assert(pad == 0);
        assert(dimsize[index] == 2);

        sprintf(_newvar, "(%c//%d)", 'a' + index, in_dim - dels);
        std::string newvar1 = _newvar;
        sprintf(_newvar, "(%c%%%d+(%c//%d)*%d)", 'a' + index, in_dim - dels,
                'a' + index, in_dim - dels, dels);
        std::string newvar2 = _newvar;
        dfsFuse(tree, newvar1, newvar2);

        dimsize[index] = dimsize[index] * (dimsize[index + 1] - dels);
        dimsize.erase(dimsize.begin() + index + 1);
        return tree;
    }
    sprintf(_newvar, "(%c//%d)", 'a' + index, in_dim + pad);
    std::string newvar1 = _newvar;
    sprintf(_newvar, "(%c%%%d)", 'a' + index, in_dim + pad);
    std::string newvar2 = _newvar;
    dfsFuse(tree, newvar1, newvar2);

    std::vector<std::string> condition;
    condition.reserve(1);
    char newCond[100];
    sprintf(newCond, "(%c%%%d) < %d", 'a' + index, in_dim + pad, in_dim);
    std::string cond = newCond;
    condition.push_back(cond);
    tpm::DecisionTree *newNode = new tpm::DecisionTree(condition, tree);

    dimsize[index] = dimsize[index] * (dimsize[index + 1] + pad) - pad;
    dimsize.erase(dimsize.begin() + index + 1);
    return newNode;
}

void tpm::TransFuse::dfsFuse(tpm::DecisionTree *tree, std::string newvar1,
                             std::string newvar2) {
    if (tree == nullptr)
        return;
    std::vector<std::string> mapLambda = tree->getCond();
    for (auto &exp : mapLambda) {
        std::string newexp = "";
        for (auto c : exp) {
            if (isVariable(c) && getVarIndex(c) > index + 1)
                newexp += (c - 1);
            else if (isVariable(c) && getVarIndex(c) == index + 1)
                newexp += newvar2;
            else if (isVariable(c) && getVarIndex(c) == index)
                newexp += newvar1;
            else
                newexp += c;
        }
        exp = newexp;
    }
    tree->putCond(mapLambda);
    dfsFuse(tree->getThenNode(), newvar1, newvar2);
    dfsFuse(tree->getElseNode(), newvar1, newvar2);
}

void tpm::TransFuse::print() {
    std::cout << "TransFuse(index=" << index << ")" << std::endl;
}

tpm::DecisionTree *tpm::TransReorder::putTranspose(tpm::DecisionTree *tree,
                                                   std::vector<int> &dimsize) {
    assert(dimsize.size() == permu.size());
    std::vector<int> rela(permu.size(), -1);
    std::vector<int> newdimsize;
    newdimsize.reserve(permu.size());
    for (size_t i = 0, iEnd = permu.size(); i < iEnd; ++i) {
        assert(rela[permu[i]] == -1);
        rela[permu[i]] = i;
        newdimsize.push_back(dimsize[permu[i]]);
    }
    dimsize = newdimsize;
    dfsReorder(tree, rela);
    return tree;
}

void tpm::TransReorder::dfsReorder(tpm::DecisionTree *tree,
                                   std::vector<int> &rela) {
    if (tree == nullptr)
        return;
    std::vector<std::string> mapLambda = tree->getCond();
    for (auto &exp : mapLambda) {
        for (auto &c : exp) {
            if (isVariable(c))
                c = getVarName(rela[getVarIndex(c)]);
        }
    }
    tree->putCond(mapLambda);
    dfsReorder(tree->getThenNode(), rela);
    dfsReorder(tree->getElseNode(), rela);
}

void tpm::TransReorder::print() {
    std::cout << "TransReorder(perm=[";
    for (size_t i = 0, iEnd = permu.size(); i < iEnd; ++i)
        std::cout << permu[i] << (i == permu.size() - 1 ? "" : ", ");
    std::cout << "])" << std::endl;
}

std::string tpm::TransposeEngine::getLambdaExp(tpm::DecisionTree *tree) {
    std::string condCon = "";
    std::vector<std::string> mapLambda = tree->getCond();
    for (auto exp : mapLambda) {
        condCon += exp + ",";
    }
    condCon.pop_back();
    if (tree->getThenNode() == nullptr)
        return "I[0][" + condCon + "]";
    std::string thenLam = getLambdaExp(tree->getThenNode());
    std::string elseLam = "tvm.tir.const(0., \"float32\")";
    if (tree->getElseNode() != nullptr)
        elseLam = getLambdaExp(tree->getElseNode());
    return "tvm.tir.if_then_else(" + condCon + ", " + thenLam + ", " + elseLam +
           ")";
}

std::string tpm::TransposeEngine::getLambda(std::vector<TransBasic *> oper,
                                            const std::vector<int> &inDims) {
    int nDim = inDims.size();

    std::vector<std::string> mapLambda;
    mapLambda.reserve(nDim);
    for (int i = 0; i < nDim; ++i) {
        std::string var = " ";
        var[0] = getVarName(i);
        mapLambda.push_back(var);
    }
    tpm::DecisionTree *tree = new tpm::DecisionTree(mapLambda);
    auto outDims = inDims;
    for (auto op : oper) {
        tree = op->putTranspose(tree, outDims);
    }
    nDim = outDims.size();
    std::string lambda = "lambda ";
    for (int i = 0; i < nDim; ++i) {
        std::string var = " ,";
        var[0] = getVarName(i);
        lambda += var;
    }
    lambda.pop_back();
    lambda += ": ";
    lambda += getLambdaExp(tree);

    return lambda;
}

void TransSplit::getOptypeDim(std::string &optype, std::vector<int> &dim,
                              std::vector<int> &extra) {
    // TODO: add operator with padding if required
    optype = "Reshape";
    std::vector<int> ndim;
    for (size_t i = 0, iEnd = dim.size(); i < iEnd; ++i) {
        if ((int)i == index) {
            ndim.emplace_back(dim[i] / in_dim);
            ndim.emplace_back(in_dim);
            extra.emplace_back(-1);
            extra.emplace_back(in_dim);
        } else {
            ndim.emplace_back(dim[i]);
            extra.emplace_back(dim[i]);
        }
    }
    dim = ndim;
}

void TransFuse::getOptypeDim(std::string &optype, std::vector<int> &dim,
                             std::vector<int> &extra) {
    optype = "Reshape";
    std::vector<int> ndim;
    for (size_t i = 0, iEnd = dim.size(); i < iEnd; ++i) {
        if ((int)i == index) {
            ndim.emplace_back(dim[i] * dim[i + 1]);
            extra.emplace_back(dim[i] * dim[i + 1]);
            i++;
        } else {
            ndim.emplace_back(dim[i]);
            extra.emplace_back(dim[i]);
        }
    }
    dim = ndim;
}

void TransReorder::getOptypeDim(std::string &optype, std::vector<int> &dim,
                                std::vector<int> &extra) {
    optype = "Transpose";
    std::vector<int> ndim = dim;
    for (size_t i = 0, iEnd = permu.size(); i < iEnd; ++i) {
        ndim[i] = dim[permu[i]];
    }
    dim = ndim;
    extra = permu;
}

} // namespace tpm
