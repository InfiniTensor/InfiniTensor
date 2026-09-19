#include "operators/resize.h"
#include <algorithm>
#include <cmath>
namespace infini {
ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor sizes,
                     Tensor scales, Tensor roi,
                     EKeepAspectRatioPolicy ratioPolicy,
                     ENearestMode nearestMode,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input}, {output}), coMode(coordTransMode),
      mode(ECoeffMode::nearest), nearestMode(nearestMode),
      ratioPolicy(ratioPolicy) {
    init(input, sizes, scales, roi, axes);
    IT_ASSERT(checkValid(graph));
}

ResizeObj::ResizeObj(GraphObj *graph, Tensor input, Tensor output,
                     const std::optional<vector<int>> &axes, Tensor sizes,
                     Tensor scales, Tensor roi, ECoeffMode mode,
                     EKeepAspectRatioPolicy ratioPolicy,
                     ECoordinateTransMode coordTransMode)
    : OperatorObj(OpType::Resize, {input}, {output}), coMode(coordTransMode),
      mode(mode), nearestMode(ENearestMode::none), ratioPolicy(ratioPolicy) {
    init(input, sizes, scales, roi, axes);
    IT_ASSERT(checkValid(graph));
}

void ResizeObj::init(const Tensor &input, const Tensor &sizes,
                     const Tensor &scales, const Tensor &roi,
                     const std::optional<vector<int>> &axes) {
    IT_ASSERT(!(nullptr != sizes && nullptr != scales));

    // inputs of operator must not be nullptr, due to the check in
    // OperatorObj::OperatorObj
    if (nullptr != sizes) {
        setGivenSizes(true);
        inputs.push_back(sizes);
        InitBySizes(input, sizes, axes);
    } else if (nullptr != scales) {
        setGivenSizes(false);
        inputs.push_back(scales);
        InitByScales(input, scales, axes);
    }

    // roi
    if (ECoordinateTransMode::tfCropAndResize == coMode) {
        IT_ASSERT(nullptr != roi);
        inputs.push_back(roi);
        IT_ASSERT(roi->getRank() == 1);
        IT_ASSERT((size_t)roi->getDims()[0] == this->axes.size() * 2);

        // init roi_start = 0;roi_end =1
        size_t nDims = input->getRank();
        for (size_t i = 0; i < nDims; ++i) {
            this->roi.emplace_back(0);
        }
        for (size_t i = 0; i < nDims; ++i) {
            this->roi.emplace_back(1);
        }

        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        std::shared_ptr<float> dataObj((float *)runtime->alloc(roi->getBytes()),
                                       [&](float *p) { runtime->dealloc(p); });
        auto data = dataObj.get();
        roi->getRuntime()->copyBlobToCPU(
            (void *)data, roi->getRawDataPtr<void *>(), roi->getBytes());

        for (size_t i = 0; i < this->axes.size(); ++i) {
            this->roi[this->axes[i]] = data[i];
            this->roi[this->axes[i] + nDims] = data[i + this->axes.size()];
        }
    }
}

void ResizeObj::InitBySizes(Tensor input, Tensor sizes,
                            const std::optional<vector<int>> &axes) {
    IT_ASSERT(sizes != nullptr);
    size_t size = sizes->getDims()[0];
    IT_ASSERT(size == input->getRank() ||
              (axes != std::nullopt && size == (*axes).size()));

    if (axes == std::nullopt) {
        for (size_t i = 0; i < input->getRank(); ++i) {
            this->axes.emplace_back(i);
        }
    } else {
        // check axes
        for (size_t i = 0; i < (*axes).size(); ++i) {
            auto val = (*axes)[i];
            if (val < 0) {
                IT_TODO_HALT();
            }
            IT_ASSERT((size_t)val < inputs[0]->getRank());
            this->axes.emplace_back(val);
        }
    }
    // init this->scales
    for (size_t i = 0; i < input->getRank(); ++i) {
        this->scales.emplace_back(1);
    }

    // copy sizes data to host.
    IT_ASSERT(sizes->getDataBlob() != nullptr);
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    std::shared_ptr<int64_t> dataObj(
        (int64_t *)runtime->alloc(sizes->getBytes()),
        [&](int64_t *p) { runtime->dealloc(p); });
    auto data = dataObj.get();
    sizes->getRuntime()->copyBlobToCPU(
        (void *)data, sizes->getRawDataPtr<void *>(), sizes->getBytes());

    sizesRequested.assign(data, data + this->axes.size());
    takeScalesFromSizes(input);
}

void ResizeObj::InitByScales(Tensor input, Tensor scales,
                             const std::optional<vector<int>> &axes) {
    IT_ASSERT(scales != nullptr);
    size_t size = scales->getDims()[0];
    IT_ASSERT(size == input->getRank() ||
              (axes != std::nullopt && size == (*axes).size()));

    // copy scales data to host.
    IT_ASSERT(scales->getDataBlob() != nullptr);
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    std::shared_ptr<float> dataObj((float *)runtime->alloc(scales->getBytes()),
                                   [&](float *p) { runtime->dealloc(p); });
    auto data = dataObj.get();
    scales->getRuntime()->copyBlobToCPU(
        (void *)data, scales->getRawDataPtr<void *>(), scales->getBytes());

    // init this->scales
    for (size_t i = 0; i < input->getRank(); ++i) {
        this->scales.emplace_back(1);
    }

    if (axes == std::nullopt) {
        for (size_t i = 0; i < input->getRank(); ++i) {
            this->axes.emplace_back(i);
            IT_ASSERT(data[i] > 0);
            this->scales[i] = data[i];
        }
    } else {
        // check axes
        for (size_t i = 0; i < (*axes).size(); ++i) {
            auto val = (*axes)[i];
            if (val < 0) {
                IT_TODO_HALT();
            }
            IT_ASSERT((size_t)val < inputs[0]->getRank());
            this->axes.emplace_back(val);
            IT_ASSERT(data[i] > 0);
            this->scales[val] = data[i];
        }
    }
}

void ResizeObj::takeScalesFromSizes(const Tensor &input) {
    IT_ASSERT(isResizeBySizes());
    IT_ASSERT(sizesRequested.size() == axes.size());
    const auto inDims = input->getDims();
    const int n = static_cast<int>(axes.size());
    const auto ratio = [&](int i) {
        return (float)sizesRequested[i] / (float)inDims[axes[i]];
    };
    switch (ratioPolicy) {
    case EKeepAspectRatioPolicy::stretch:
        // Each axis reaches the size asked of it on its own, so each keeps its
        // own ratio.
        for (int i = 0; i < n; ++i) {
            scales[axes[i]] = ratio(i);
        }
        break;
    case EKeepAspectRatioPolicy::notLarger: {
        // One ratio for every axis, the smallest, so that none overshoots the
        // size asked of it.
        float scale = ratio(0);
        for (int i = 1; i < n; ++i) {
            scale = std::min(scale, ratio(i));
        }
        for (int i = 0; i < n; ++i) {
            scales[axes[i]] = scale;
        }
        break;
    }
    case EKeepAspectRatioPolicy::notSmaller: {
        // The largest, so that none falls short of it.
        float scale = ratio(0);
        for (int i = 1; i < n; ++i) {
            scale = std::max(scale, ratio(i));
        }
        for (int i = 0; i < n; ++i) {
            scales[axes[i]] = scale;
        }
        break;
    }
    default:
        IT_ASSERT(0);
    }
}

vector<DataType> ResizeObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    if (inputs.size() == 3) {
        auto roi = inputs[2];
        IT_ASSERT(roi && roi->getDType() == DataType::Float32);
    }
    if (isResizeBySizes()) {
        auto sizes = inputs[1];
        IT_ASSERT(sizes && sizes->getDType() == DataType::Int64);
    } else {
        auto scales = inputs[1];
        IT_ASSERT(scales && scales->getDType() == DataType::Float32);
    }
    return {inputs[0]->getDType()};
}

float ResizeObj::round_int(float x) const {
    return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

// output shape is related to sizes/scales value.
optional<vector<Shape>> ResizeObj::inferShape(const TensorVec &inputs) {
    // A ratio taken against a placeholder input dimension describes that
    // placeholder and nothing else, so the scales are worked out again from the
    // sizes the model asked for whenever the input shape may have moved.
    if (isResizeBySizes()) {
        takeScalesFromSizes(inputs[0]);
    }
    auto inDims = inputs[0]->getDims();
    Shape ret = inDims;
    int rank = inputs[0]->getRank();
    for (int i = 0; i < rank; ++i) {
        int size = round_int(scales[i] * inDims[i]);
        ret[i] = size;
    }

    return {{ret}};
}

vector<DimSource> ResizeObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    IT_ASSERT(dim < outputs[0]->getRank());
    const bool resized = std::find(axes.begin(), axes.end(),
                                   static_cast<int>(dim)) != axes.end();
    // An axis nobody resized is passed through, and one resized by a scale is
    // that many times the dimension it came from: either way it follows its own
    // place in the input.
    if (!resized || !isResizeBySizes()) {
        return {DimSource{0, dim}};
    }
    // Resizing by sizes settles the axis at the size the model asked for, which
    // no input shape can change -- except under a policy that holds one ratio
    // across the resized axes, where the ratio chosen depends on all of them
    // and so every one of those dimensions is followed.
    if (ratioPolicy == EKeepAspectRatioPolicy::stretch) {
        return {};
    }
    vector<DimSource> sources;
    sources.reserve(axes.size());
    for (const auto axis : axes) {
        sources.push_back(DimSource{0, static_cast<size_t>(axis)});
    }
    return sources;
}

std::string ResizeObj::toString() const {
    std::ostringstream os;
    os << "Resize"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    if (inputs.size() == 3) {
        os << "roi=" << vecToString(inputs[2]->getDims()) << ",";
    }
    if (isResizeBySizes()) {
        os << "sizes=" << vecToString(inputs[1]->getDims()) << ",";
    } else {
        os << "scales=" << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "axes=" << vecToString(axes) << ",";
    os << "coMode=" << enum_to_underlying(coMode) << ",";
    os << "nearestMode=" << enum_to_underlying(nearestMode) << ",";
    os << "ratioPolicy=" << enum_to_underlying(ratioPolicy) << ",";

    os << "input=" << inputs[0]->getGuid() << ",";
    os << inputs[1]->getGuid() << ",";
    if (inputs.size() == 3) {
        os << inputs[2]->getGuid() << ",";
    }
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ResizeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    for (size_t i = 0; i < outputs[0]->getRank(); ++i) {
        ret.emplace_back(outputs[0]->getDims()[i]);
    }
    // ratioPolicy only effects output shape, so did not need
    // here.
    ret.emplace_back(enum_to_underlying(coMode));
    ret.emplace_back(enum_to_underlying(nearestMode));
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ResizeObj::getOpAttrVector() const {
    vector<int> ret = axes;
    ret.emplace_back(enum_to_underlying(coMode));
    ret.emplace_back(enum_to_underlying(nearestMode));
    ret.emplace_back(enum_to_underlying(ratioPolicy));
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

} // namespace infini
