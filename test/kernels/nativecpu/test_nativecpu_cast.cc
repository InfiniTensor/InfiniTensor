#include "core/graph.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

/// One conversion, given as the values going in and the values expected out.
template <typename From, typename To>
void testCast(DataType fromType, CastType castType, const vector<From> &input,
              const vector<To> &expected) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    const auto in =
        g->addTensor(Shape{static_cast<int>(input.size())}, fromType);
    const auto op = g->addOp<CastObj>(in, nullptr, castType);
    g->dataMalloc();
    in->copyin(input);

    runtime->run(g);
    EXPECT_TRUE(op->getOutput()->equalData(expected));
}

TEST(Cast, NativeCpu) {
    // The conversion an attention export asks for: a dimension read at
    // runtime, turned into a scale factor.
    testCast<int64_t, float>(DataType::Int64, CastType::Int642Float,
                             {8, 0, -3, 1024}, {8.0f, 0.0f, -3.0f, 1024.0f});
    // A quotient is truncated towards zero rather than rounded.
    testCast<float, int32_t>(DataType::Float32, CastType::Float2Int32,
                             {2.9f, -2.9f, 0.4f}, {2, -2, 0});
    testCast<int32_t, int64_t>(DataType::Int32, CastType::Int322Int64,
                               {7, -7, 0}, {7, -7, 0});
    testCast<int64_t, int32_t>(DataType::Int64, CastType::Int642Int32,
                               {7, -7, 0}, {7, -7, 0});
    // A cast to the same type is a copy.
    testCast<float, float>(DataType::Float32, CastType::Float2Float,
                           {1.5f, -1.5f}, {1.5f, -1.5f});
}

} // namespace infini
