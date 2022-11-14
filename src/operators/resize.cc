#include "operators/resize.h"
#include <cmath>
namespace infini {
ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor sizes,
                     EKeepAspectRatioPolicy ratioPolicy,
                     ENearestMode nearestMode,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input, nullptr, nullptr, sizes}, {output}),
      coMode(coordTransMode), mode(ECoeffMode::nearest),
      nearestMode(nearestMode), ratioPolicy(ratioPolicy) {
    if (coordTransMode == ECoordinateTransMode::tfCropAndResize)
        IT_TODO_HALT();
    InitBySizes(input, sizes, axes);

    IT_ASSERT(checkValid(graph));
}

ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor scales,
                     ENearestMode nearestMode,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input, nullptr, scales, nullptr}, {output}),
      coMode(coordTransMode), mode(ECoeffMode::nearest),
      nearestMode(nearestMode) {
    InitByScales(input, scales, axes);

    IT_ASSERT(checkValid(graph));
}

ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor sizes,
                     EKeepAspectRatioPolicy ratioPolicy, ECoeffMode mode,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input, nullptr, nullptr, sizes}, {output}),
      coMode(coordTransMode), mode(mode), ratioPolicy(ratioPolicy) {
    if (coordTransMode == ECoordinateTransMode::tfCropAndResize)
        IT_TODO_HALT();
    InitBySizes(input, sizes, axes);

    IT_ASSERT(checkValid(graph));
}

ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor scales,
                     ECoeffMode mode, ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input, nullptr, scales, nullptr}, {output}),
      coMode(coordTransMode), mode(mode) {
    if (coordTransMode == ECoordinateTransMode::tfCropAndResize)
        IT_TODO_HALT();
    InitByScales(input, scales, axes);

    IT_ASSERT(checkValid(graph));
}

void ResizeObj::InitBySizes(Tensor input, Tensor sizes,
                            const std::optional<vector<int>> &axes) {
    IT_ASSERT(sizes != nullptr);
    size_t size = sizes->getDims()[0];
    IT_ASSERT(size == input->getDims().size() ||
              (axes != std::nullopt && size == (*axes).size()));

    if (axes == std::nullopt)
        for (size_t i = 0; i < input->getDims().size(); ++i)
            this->axes.emplace_back(i);
    else
        // check axes
        for (size_t i = 0; i < (*axes).size(); ++i) {
            auto val = (*axes)[i];
            if (val < 0)
                IT_TODO_HALT();
            IT_ASSERT((size_t)val < inputs[0]->getDims().size());
            this->axes.emplace_back(val);
        }

    // init this->scales
    for (size_t i = 0; i < input->getDims().size(); ++i) {
        this->scales.emplace_back(1);
    }

    // copy sizes data to host.
    IT_ASSERT(sizes->getDataBlob() != nullptr);
    Runtime runtime = CpuRuntimeObj::getInstance();
    int *data = (int *)runtime->alloc(sizes->getBytes());
    sizes->getRuntime()->copyBlobToCPU(
        (void *)data, sizes->getRawDataPtr<void *>(), sizes->getBytes());

    auto inDims = input->getDims();
    int n = this->axes.size();
    switch (ratioPolicy) {
    case EKeepAspectRatioPolicy::stretch:
        for (int i = 0; i < n; ++i)
            scales[this->axes[i]] =
                (float)data[i] / (float)inDims[this->axes[i]];
        break;
    case EKeepAspectRatioPolicy::notLarger: {
        float scale = (float)data[0] / (float)inDims[this->axes[0]];
        for (int i = 1; i < n; ++i) {
            auto tmp = (float)data[i] / (float)inDims[this->axes[i]];
            scale = scale < tmp ? scale : tmp;
        }
        for (int i = 0; i < n; ++i)
            scales[this->axes[i]] = scale;
        break;
    }
    case EKeepAspectRatioPolicy::notSmaller: {
        float scale = (float)data[0] / (float)inDims[this->axes[0]];
        for (int i = 1; i < n; ++i) {
            auto tmp = (float)data[i] / (float)inDims[this->axes[i]];
            scale = scale > tmp ? scale : tmp;
        }
        for (int i = 0; i < n; ++i)
            scales[this->axes[i]] = scale;
        break;
    }
    default:
        IT_ASSERT(0);
    }

    runtime->dealloc(data);
}

void ResizeObj::InitByScales(Tensor input, Tensor scales,
                             const std::optional<vector<int>> &axes) {
    IT_ASSERT(scales != nullptr);
    size_t size = scales->getDims()[0];
    IT_ASSERT(size == input->getDims().size() ||
              (axes != std::nullopt && size == (*axes).size()));

    // copy scales data to host.
    IT_ASSERT(scales->getDataBlob() != nullptr);
    Runtime runtime = CpuRuntimeObj::getInstance();
    float *data = (float *)runtime->alloc(scales->getBytes());
    scales->getRuntime()->copyBlobToCPU(
        (void *)data, scales->getRawDataPtr<void *>(), scales->getBytes());

    // init this->scales
    for (size_t i = 0; i < input->getDims().size(); ++i) {
        this->scales.emplace_back(1);
    }

    if (axes == std::nullopt)
        for (size_t i = 0; i < input->getDims().size(); ++i) {
            this->axes.emplace_back(i);
            IT_ASSERT(data[i] > 0);
            this->scales[i] = data[i];
        }
    else
        // check axes
        for (size_t i = 0; i < (*axes).size(); ++i) {
            auto val = (*axes)[i];
            if (val < 0)
                IT_TODO_HALT();
            IT_ASSERT((size_t)val < inputs[0]->getDims().size());
            this->axes.emplace_back(val);
            IT_ASSERT(data[i] > 0);
            this->scales[val] = data[i];
        }

    runtime->dealloc(data);
}

vector<DataType> ResizeObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 4);
    auto roi = inputs[1];
    auto scales = inputs[2];
    auto sizes = inputs[3];
    IT_ASSERT(roi == nullptr || roi->getDType() == DataType::Float32);
    IT_ASSERT(scales == nullptr || scales->getDType() == DataType::Float32);
    IT_ASSERT(sizes == nullptr || sizes->getDType() == DataType::UInt32);
    return {inputs[0]->getDType()};
}

bool ResizeObj::checkCoordinateTransValid(int resizedX, int origiX) const {
    if (ECoordinateTransMode::alignCorners == coMode) {
        return (!(resizedX <= 1 && origiX != resizedX));
    }
    return true;
}

float ResizeObj::round_int(float x) const {
    return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

// output shape is related to sizes/scales value.
optional<vector<Shape>> ResizeObj::inferShape(const TensorVec &inputs) const {
    auto inDims = inputs[0]->getDims();
    Shape ret = inDims;
    int nDim = inDims.size();
    for (int i = 0; i < nDim; ++i) {
        int size = round_int(scales[i] * inDims[i]);
        IT_ASSERT(checkCoordinateTransValid(size, inDims[i]));
        ret[i] = size;
    }

    return {{ret}};
}

std::string ResizeObj::toString() const {
    std::ostringstream os;
    os << "Resize"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    if (inputs[1] != nullptr)
        os << "roi=" << vecToString(inputs[1]->getDims()) << ",";
    if (inputs[2] != nullptr)
        os << "scales=" << vecToString(inputs[2]->getDims()) << ",";
    if (inputs[3] != nullptr)
        os << "sizes=" << vecToString(inputs[3]->getDims()) << ",";
    os << "axes=" << vecToString(axes) << ",";
    os << "coMode=" << enum_to_underlying(coMode) << ",";
    os << "nearestMode=" << enum_to_underlying(nearestMode) << ",";
    os << "ratioPolicy=" << enum_to_underlying(ratioPolicy) << ",";

    os << "input=" << inputs[0]->getGuid() << ",";
    if (inputs[1] != nullptr)
        os << inputs[1]->getGuid() << ",";
    if (inputs[2] != nullptr)
        os << inputs[2]->getGuid() << ",";
    if (inputs[3] != nullptr)
        os << inputs[3]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ResizeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    for (size_t i = 0; i < outputs[0]->getDims().size(); ++i)
        ret.emplace_back(outputs[0]->getDims()[i]);
    // ratioPolicy only effects output shape, so did not need
    // here.
    ret.emplace_back(enum_to_underlying(coMode));
    ret.emplace_back(enum_to_underlying(nearestMode));
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> ResizeObj::getOpAttrVector() const {
    vector<int> ret = axes;
    ret.emplace_back(enum_to_underlying(coMode));
    ret.emplace_back(enum_to_underlying(nearestMode));
    ret.emplace_back(enum_to_underlying(ratioPolicy));
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

} // namespace infini
