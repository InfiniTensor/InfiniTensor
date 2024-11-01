#include "code_gen/operator.h"
#include "code_gen/tensor.h"
#include <iostream>
#include "test.h"

using namespace tpm;

ssize_t getOffset(std::vector<ssize_t> index, std::vector<int> shape) {
    ssize_t ret = index[0];
    for (size_t i = 1; i < index.size(); ++i)
        ret = ret * shape[i] + index[i];
    return ret;
}

void tconv_compute_test1() {
    const int sh = 2, sw = 2;
    const int ph = 1, pw = 1;
    const int dh = 1, dw = 1;

    const int n = 1, c = 2, h = 2, w = 2;
    const int f = 3, r = 4, s = 4;

    Tensor input((Dim){n, h, w, f});  // {1, 2, 2, 3}
    Tensor weight((Dim){r, s, f, c}); // {4, 4, 3, 2}

    const int input_len = n * h * w * f;
    VType *_input = new VType[input_len];

    int weight_len = r * s * f * c;
    VType *_weight = new VType[weight_len];

    for (int nn = 0; nn < n; ++nn)
        for (int hh = 0; hh < h; ++hh)
            for (int ww = 0; ww < w; ++ww)
                for (int ff = 0; ff < f; ++ff) {
                    // Assign value from 0 to nfhw
                    // X[n, h, w, f] = ((n * H + h) * W + w) * F + f;
                    _input[getOffset({nn, hh, ww, ff}, {n, h, w, f})] =
                        getOffset({nn, hh, ww, ff}, {n, h, w, f});
                }
    for (int rr = 0; rr < r; ++rr)
        for (int ss = 0; ss < s; ++ss)
            for (int ff = 0; ff < f; ++ff)
                for (int cc = 0; cc < c; ++cc) {
                    // Assign value from 0 to frsc
                    // K[f, c, r, s] = ((f * R + r) * S + s) * C + c;
                    _weight[getOffset({rr, ss, ff, cc}, {r, s, f, c})] =
                        getOffset({ff, rr, ss, cc}, {f, r, s, c});
                }

    assert(input.dataMalloc());
    assert(input.setData(_input));
    assert(weight.dataMalloc());
    assert(weight.setData(_weight)); // {1, 2, 4, 4}

    input.print();
    weight.print();

    ConvTransOp convTransOp(&input, &weight, ph, pw, sh, sw, dh, dw);
    Tensor *output = convTransOp.compute();
    output->print();
    // Pytorch
    // tensor([[[[ 190.,  740.,  770.,  592.],
    //       [ 992., 2704., 2836., 1832.],
    //       [1184., 3232., 3364., 2168.],
    //       [1114., 2660., 2762., 1624.]],

    //      [[ 193.,  755.,  785.,  604.],
    //       [1016., 2770., 2902., 1874.],
    //       [1208., 3298., 3430., 2210.],
    //       [1135., 2711., 2813., 1654.]]]])torch.Size([1, 2, 4, 4])
    const std::vector<unsigned int> ans = {
        190,  740,  770,  592,  992,  2704, 2836, 1832, 1184, 3232, 3364,
        2168, 1114, 2660, 2762, 1624, 193,  755,  785,  604,  1016, 2770,
        2902, 1874, 1208, 3298, 3430, 2210, 1135, 2711, 2813, 1654};
    EXPECT_TRUE(ans.size() == output->size());
    for (size_t i = 0; i < ans.size(); ++i) {
        std::cout<<"test"<<ans[i]<<" "<<output->getData(i)<<std::endl;
        EXPECT_TRUE(ans[i] == output->getData(i));
    }
}

void tconv_compute_test2() {
    const int sh = 1, sw = 1;
    const int ph = 0, pw = 0;
    const int dh = 1, dw = 1;

    const int n = 1, c = 2, h = 1, w = 1;
    const int f = 3, r = 2, s = 2;

    Tensor input((Dim){n, h, w, f});  // {1, 1, 1, 3}
    Tensor weight((Dim){r, s, f, c}); // {2, 2, 3, 2}

    const int input_len = n * h * w * f;
    VType *_input = new VType[input_len];

    int weight_len = r * s * f * c;
    VType *_weight = new VType[weight_len];

    for (int nn = 0; nn < n; ++nn)
        for (int hh = 0; hh < h; ++hh)
            for (int ww = 0; ww < w; ++ww)
                for (int ff = 0; ff < f; ++ff) {
                    // Assign value from 0 to nfhw
                    // X[n, h, w, f] = ((n * H + h) * W + w) * F + f;
                    _input[getOffset({nn, hh, ww, ff}, {n, h, w, f})] =
                        getOffset({nn, hh, ww, ff}, {n, h, w, f});
                }
    for (int rr = 0; rr < r; ++rr)
        for (int ss = 0; ss < s; ++ss)
            for (int ff = 0; ff < f; ++ff)
                for (int cc = 0; cc < c; ++cc) {
                    // Assign value from 0 to frsc
                    // K[r, s, f, c] = ((f * R + r) * S + s) * C + c;
                    _weight[getOffset({rr, ss, ff, cc}, {r, s, f, c})] =
                        getOffset({ff, rr, ss, cc}, {f, r, s, c});
                }

    assert(input.dataMalloc());
    assert(input.setData(_input));
    assert(weight.dataMalloc());
    assert(weight.setData(_weight));

    input.print();
    weight.print();

    ConvTransOp convTransOp(&input, &weight, ph, pw, sh, sw, dh, dw);
    Tensor *output = convTransOp.compute();
    output->print();
    // Pytorch
    // tensor([[[[40., 46.],
    //           [52., 58.]],
    //          [[43., 49.],
    //           [55., 61.]]]]), torch.Size([1, 2, 2, 2])
    const std::vector<unsigned int> ans = {40, 46, 52, 58, 43, 49, 55, 61};
    EXPECT_TRUE(ans.size() == output->size());
    for (size_t i = 0; i < ans.size(); ++i)
        EXPECT_TRUE(ans[i] == output->getData(i));
}

TEST(CONVTRANS_TEST_1, Cuda_codeGenerate) {
    printf("---------- Test 1 ----------\n");
    tconv_compute_test1();
    printf("---------- Success ----------\n");
    printf("---------- Test 2 ----------\n");
    tconv_compute_test2();
    printf("---------- Success ----------\n");
}