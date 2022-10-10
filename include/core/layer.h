#pragma once
#include "core/tensor.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/kernel.h"

namespace infini {
    class Convolution {
        private:
            Tensor input;
            Tensor weight;
            Tensor output;
            Tensor dInput;
            Tensor dWeight;
            Tensor dOutput;
        public:
            Convolution(Tensor input_, int pad, int window, int stride, int num); 
            Tensor forward();
            Tensor backward();
    };
}
