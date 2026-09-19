#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {

namespace {
/// Multiplies the two given operands, with the given bias if there is one, and
/// checks the result against one worked out by hand. Small whole numbers are
/// used throughout so that every expected element can be read off the operands.
void runMatmul(const Shape &shapeA, const vector<float> &a,
               const Shape &shapeB, const vector<float> &b,
               const Shape &shapeBias, const vector<float> &biasData,
               const Shape &expectedShape, const vector<float> &expected) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType::Float32);
    auto B = g->addTensor(shapeB, DataType::Float32);
    Tensor bias = shapeBias.empty() && biasData.empty()
                      ? nullptr
                      : g->addTensor(shapeBias, DataType::Float32);
    auto op = g->addOp<MatmulObj>(A, B, nullptr, false, false, bias);
    g->dataMalloc();
    A->copyin(a);
    B->copyin(b);
    if (bias) {
        bias->copyin(biasData);
    }
    runtime->run(g);

    EXPECT_EQ(op->getOutput()->getDims(), expectedShape);
    EXPECT_TRUE(op->getOutput()->equalData(expected));
}
} // namespace

TEST(MatmulNaive, PlainMatrices) {
    // [[1,2,3],[4,5,6]] times [[1,2],[3,4],[5,6]].
    runMatmul({2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {1, 2, 3, 4, 5, 6}, {}, {},
              {2, 2}, {22, 28, 49, 64});
}

TEST(MatmulNaive, BiasPerColumn) {
    // The shape a linear layer's bias has: one value for each output column,
    // added to every row alike.
    runMatmul({2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {1, 2, 3, 4, 5, 6}, {2},
              {10, 20}, {2, 2}, {32, 48, 59, 84});
}

TEST(MatmulNaive, BiasPerRow) {
    // A bias of its own for each row, which a column vector describes.
    runMatmul({2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {1, 2, 3, 4, 5, 6}, {2, 1},
              {10, 20}, {2, 2}, {32, 38, 69, 84});
}

TEST(MatmulNaive, BiasForEveryElement) {
    runMatmul({2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {1, 2, 3, 4, 5, 6}, {2, 2},
              {1, 2, 3, 4}, {2, 2}, {23, 30, 52, 68});
}

TEST(MatmulNaive, BiasOfOneValue) {
    runMatmul({2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {1, 2, 3, 4, 5, 6}, {1},
              {100}, {2, 2}, {122, 128, 149, 164});
}

TEST(MatmulNaive, BatchedOperands) {
    // Two matrices on each side, each pair multiplied with its own partner.
    runMatmul({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2},
              {1, 0, 0, 1, 2, 0, 0, 2}, {}, {}, {2, 2, 2},
              {1, 2, 3, 4, 10, 12, 14, 16});
}

TEST(MatmulNaive, OneOperandSharedAcrossTheBatch) {
    // A stack of matrices against a single one, which every matrix in the stack
    // is multiplied by. A zero stride says so, and reading the second matrix
    // from past the end of the operand is what getting that wrong looks like.
    runMatmul({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2}, {1, 0, 0, 1}, {}, {},
              {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(MatmulNaive, BiasIsSharedAcrossTheBatch) {
    // A bias belongs to the layer rather than to the batch, so the same one is
    // added to every matrix.
    runMatmul({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2}, {1, 0, 0, 1}, {2},
              {10, 20}, {2, 2, 2}, {11, 22, 13, 24, 15, 26, 17, 28});
}

TEST(MatmulNaive, BatchedWithAShapeToBroadcast) {
    // A leading dimension of one on one side broadcasts against two on the
    // other, which is a shape an exporter produces around attention.
    runMatmul({1, 2, 2}, {1, 2, 3, 4}, {2, 2, 2}, {1, 0, 0, 1, 2, 0, 0, 2}, {},
              {}, {2, 2, 2}, {1, 2, 3, 4, 2, 4, 6, 8});
}

} // namespace infini
