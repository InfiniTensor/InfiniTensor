#include "code_gen/transpose.h"
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace tpm;

int main() {
    vector<TransBasic *> oper;

    // n, c, h, w -> n/2, c, h*2+pad, w
    std::vector<int> dims({16, 32, 32, 32});
    TransBasic *op1 = new TransSplit({16, 32, 32, 32}, 0, 2);
    TransBasic *op2 = new TransReorder({0, 2, 1, 3, 4});
    TransBasic *op3 = new TransFuse({8, 32, 2, 32, 32}, 2, 1);
    oper.push_back(op1);
    oper.push_back(op2);
    oper.push_back(op3);

    auto &worker = TransposeEngine::getInstance();
    cout << worker.getKernelTime(oper, dims) << endl;
    return 0;
}
