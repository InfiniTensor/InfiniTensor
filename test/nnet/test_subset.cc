#include "nnet/dlt.h"
#include "nnet/expr.h"
#include "nnet/permutation.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(Subset, Simple) {
    SubsetGenerator<string> gen{{"a", "b", "c"}};
    int cnt = 0;
    do {
        ++cnt;
        if (cnt == 1) {
            EXPECT_EQ(gen.get()[0], "a");
        }
    } while (gen.next());
    EXPECT_EQ(cnt, 8 - 2);
}