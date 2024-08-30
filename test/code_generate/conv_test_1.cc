#include "code_gen/operator.h"
#include "code_gen/tensor.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace tpm;

int main() {
    int ph = 2, pw = 2;
    int sh = 1, sw = 1;
    int dh = 2, dw = 2;

    int n = 16, c = 16, h = 32, w = 32;
    int f = 32, r = 3, s = 3;

    Tensor input((Dim){n, c, h, w});
    Tensor weight((Dim){f, c, r, s});

    FILE *in = fopen("dat/conv_test/input.dat", "r");
    int input_len = n * c * h * w;
    VType *_input = new VType[input_len];
    for (int i = 0; i < input_len; ++i)
        if (!fscanf(in, "%u", &_input[i]))
            return -1;
    fclose(in);

    FILE *wgt = fopen("dat/conv_test/weight.dat", "r");
    int weight_len = f * c * r * s;
    VType *_weight = new VType[weight_len];
    for (int i = 0; i < weight_len; ++i)
        if (!fscanf(wgt, "%u", &_weight[i]))
            return -1;
    fclose(wgt);

    assert(input.dataMalloc());
    assert(input.setData(_input));
    assert(weight.dataMalloc());
    assert(weight.setData(_weight));

    ConvOp convop(&input, &weight, ph, pw, sh, sw, dh, dw);
    Tensor *outp = convop.compute();
    int output_len = outp->size();
    FILE *out = fopen("dat/conv_test/output_tpm.dat", "w");
    for (int i = 0; i < output_len; ++i)
        if (!fprintf(out, "%u ", outp->getData(i)))
            return -1;
    fclose(out);

    return 0;
}
