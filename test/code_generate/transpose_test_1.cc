#include "code_gen/operator.h"
#include "code_gen/tensor.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace tpm;

int main() {
    int n = 16, c = 16, h = 32, w = 32;

    Tensor input((Dim){n, c, h, w});
    Tensor output((Dim){n / 2, c, h * 2, w});

    FILE *in = fopen("dat/transpose_test/input.dat", "r");
    int input_len = n * c * h * w;
    VType *_input = new VType[input_len];
    for (int i = 0; i < input_len; ++i)
        if (!fscanf(in, "%u", &_input[i]))
            return -1;
    fclose(in);

    assert(input.dataMalloc());
    assert(input.setData(_input));
    assert(output.dataMalloc());

    TransposeOp transop(&input, &output, 0, {0, 1, {-1, 2}, 3});
    Tensor *outp = transop.compute();
    int output_len = outp->size();
    FILE *out = fopen("dat/transpose_test/output_tpm.dat", "w");
    for (int i = 0; i < output_len; ++i) {
        fprintf(out, "%u ", outp->getData(i));
        if (i % 32 == 31)
            fprintf(out, "\n");
    }
    fclose(out);

    return 0;
}
