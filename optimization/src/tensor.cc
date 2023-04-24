#include "tensor.h"
#include <numeric>

using namespace optimization;

Arc<Tensor> Tensor::share(Vec<size_t> shape, DataType data_type, Data data) {
    return Arc<Tensor>(
        new Tensor(std::move(shape), std::move(data_type), std::move(data)));
}

size_t Tensor::count() const {
    return shape.empty() // fmt: new line
               ? 0
               : std::accumulate(shape.begin(), shape.end(), 1,
                                 [](auto acc, auto it) { return acc * it; });
}

size_t Tensor::size() const {
    return shape.empty() // fmt: new line
               ? 0
               : std::accumulate(shape.begin(), shape.end(), data_type.size(),
                                 [](auto acc, auto it) { return acc * it; });
}

Tensor::Tensor(Vec<size_t> &&shape, DataType &&data_type, Data &&data)
    : shape(std::move(shape)),         // fmt: new line
      data_type(std::move(data_type)), //
      data(std::move(data)) {}
