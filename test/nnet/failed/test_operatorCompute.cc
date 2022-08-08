#include "graph.h"
#include "operator.h"
#include "tensor.h"
#include "gtest/gtest.h"
using namespace std;

TEST(OperatorCompute, Conv) {
    const int N = 1, C = 2, H = 2, W = 2, F = 2, R = 3, S = 3;
    auto g = new tpm::Graph();
    auto i = g->tensor({N, C, H, W});
    auto w = g->tensor({F, C, R, S});
    auto o = g->tensor({N, F, H, W});
    auto conv = g->conv(i, w, o, tpm::ConvOp::PaddingMode::Same);
    vector<tpm::VType> dataI{0, 1, 0, 0, 1, 0, 1, 0};
    vector<tpm::VType> dataW{1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1,
                             0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                             0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1};
    vector<tpm::VType> dataO{2, 1, 1, 1, 2, 0, 1, 1};
    EXPECT_EQ(i->size(), dataI.size());
    EXPECT_EQ(w->size(), dataW.size());
    EXPECT_EQ(o->size(), dataO.size());
    i->dataMalloc();
    w->dataMalloc();
    o->dataMalloc();
    i->setData(dataI.data());
    w->setData(dataW.data());
    conv->compute();
    tpm::SubGraph s(g->getOperators());
    s.print();
    for (size_t i = 0; i < dataO.size(); ++i)
        EXPECT_EQ(o->getData(i), dataO[i]);
}
// Conv[552052564]([1,2,2,2],[2,2,3,3],[0,0,0,0],[0,0,0,0],p=[1,1],s=[1,1],d=[1,1],act=0,input=0,weight=1,output=2)
// Conv[552052564]([1,2,2,2],[2,2,3,3],[0,0,0,0],[0,0,0,0],p=[1,1],s=[1,1],d=[1,1],act=0,input=0,weight=1,output=2)
