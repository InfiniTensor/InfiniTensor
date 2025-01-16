#include "code_gen/transpose.h"
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace tpm;

int main() {
    vector<TransBasic *> oper;

    // n, c, h, w -> n/2, c, h*2, w
    std::vector<int> dims({16, 512, 14, 14});
    TransBasic *op1 = new TransSplit({16, 512, 14, 14}, 2, 7);
    TransBasic *op2 = new TransReorder({0, 1, 3, 2, 4});
    TransBasic *op3 = new TransFuse({16, 512, 7, 2, 14}, 2);
    oper.push_back(op1);
    oper.push_back(op2);
    oper.push_back(op3);
    TransBasic *op4 = new TransSplit({16, 512, 14, 14}, 3, 7);
    TransBasic *op5 = new TransReorder({0, 1, 2, 4, 3});
    TransBasic *op6 = new TransFuse({16, 512, 14, 7, 2}, 3);
    oper.push_back(op4);
    oper.push_back(op5);
    oper.push_back(op6);

    auto &worker = TransposeEngine::getInstance();
    worker.getKernelTime(oper, dims);
    return 0;
}
