#include "test.h"
#include "utils/validation.h"

namespace infini {

TEST(Verify, validation) {
    int array1[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int array2[10] = {0, 1, 2, 3, 4, 5, 7, 7, 8, 11};

    int res1 = computeDifference1(array1, array2, 10);

    float array3[10] = {0.001, 0.0034, 0.44567, 1.2326, 4.5678,
                        7.657, 4.667,  8.233,   9.456,  10.334};
    float array4[10] = {0.001, 0.0033, 0.44568, 1.2324, 4.56789,
                        7.657, 4.667,  8.233,   9.456,  10.334};

    double res2 = computeDifference2(array3, array4, 10);
    double res3 = computeDifference3(array3, array4, 10);
    double res4 = computeDifference4(array3, array4, 10);
    double res5 = computeDifference5(array3, array4, 10);

    EXPECT_LE(res1, 2);
    EXPECT_LE(res2, 0.03);
    EXPECT_LE(res3, 0.03);
    EXPECT_LE(res4, 0.03);
    EXPECT_LE(res5, 0.7);
    EXPECT_GE(res5, 0.1);
}

} // namespace infini
