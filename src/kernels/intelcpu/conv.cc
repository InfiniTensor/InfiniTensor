#include "operators/conv.h"
#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
struct ConvMklPerfRecordObj : public PerfRecordObj {
    dnnl::algorithm algo = dnnl::algorithm::convolution_auto;
    void to_json(json &j) override {
        j["type"] = 1;
        j["data"] = std::make_tuple(enum_to_underlying(algo), time);
    }
    static PerfRecord from_json(const json &j) {
        ConvMklPerfRecordObj tmp;
        auto [Algo, Time] = j["data"].get<tuple<int, double>>();
        tmp.algo = (dnnl::algorithm)Algo;
        tmp.time = Time;
        return make_ref<ConvMklPerfRecordObj>(tmp);
    }
};

using ConvMklPerfRecord = Ref<ConvMklPerfRecordObj>;
class MklConv : public Kernel {
    bool createPrimitives(
        const Ref<ConvObj> &op, const ConvMklPerfRecord &record,
        const MklRuntimeObj *context, bool allowEmpty,
        std::vector<dnnl::primitive> &prims,
        std::vector<std::unordered_map<int, dnnl::memory>> &primArgs) const {
        auto srcData = op->getInputs(0)->getRawDataPtr<float *>();
        auto wData = op->getInputs(1)->getRawDataPtr<float *>();
        auto dstData = op->getOutput(0)->getRawDataPtr<float *>();

        auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const int cpg = op->getChannelPerGroup();

        auto oDims = op->getOutput(0)->getDims();
        int oH = oDims[oDims.size() - 2];
        int oW = oDims[oDims.size() - 1];

        //  create user memory that describes data layout in the buffers
        auto userSrcMd =
            dnnl::memory::desc({n, c, h, w}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nchw);
        auto userSrcMemory =
            dnnl::memory(userSrcMd, context->getEngine(), srcData);

        auto userWMd =
            dnnl::memory::desc({f, cpg, r, s}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::oihw);
        auto userWMemory = dnnl::memory(userWMd, context->getEngine(), wData);
        auto userDstMd =
            dnnl::memory::desc({n, f, oH, oW}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nchw);

        // create memory descriptors with layout tag::any, to let convolution
        // choose memory format
        // Convolution and inner product primitives choose the memory format
        // when you create them with the placeholder memory format
        // dnnl::memory::format_tag::any for input or output. The memory format
        // chosen is based on different circumstances such as hardware and
        // convolutional parameters. Using the placeholder memory format is the
        // recommended practice for convolutions, since they are the most
        // compute-intensive operations in most topologies where they are
        // present.
        auto srcMd =
            dnnl::memory::desc({n, c, h, w}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);
        auto wMd =
            dnnl::memory::desc({f, cpg, r, s}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);
        auto dstMd =
            dnnl::memory::desc({n, f, oH, oW}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);

        // create convolution descriptor
        dnnl::memory::dims strides = {sh, sw};
        dnnl::memory::dims pads = {ph, pw};
        dnnl::memory::dims dilations = {dh - 1, dw - 1};
        auto convDesc = dnnl::convolution_forward::desc(
            dnnl::prop_kind::forward_inference, record->algo, srcMd, wMd, dstMd,
            strides, dilations, pads, pads);

        dnnl::convolution_forward::primitive_desc primDesc;

        //  fused convolution
        // The non-intensive operation is added as a post-op attribute to the
        //  compute intensive primitive descriptor
        if (ActType::None != op->getAct()) {
            dnnl::algorithm algo;
            switch (op->getAct()) {
            case ActType::Relu:
                algo = dnnl::algorithm::eltwise_relu;
                break;
            case ActType::Sigmoid:
                algo = dnnl::algorithm::eltwise_logsigmoid;
                break;
            case ActType::Tanh:
                algo = dnnl::algorithm::eltwise_tanh;
                break;

            default:
                IT_ASSERT(0);
            }
            dnnl::primitive_attr attr;
            dnnl::post_ops po;
            po.append_eltwise(1.f, algo, 0.f, 0.f);
            attr.set_post_ops(po);

            primDesc = dnnl::convolution_forward::primitive_desc(
                convDesc, attr, context->getEngine(), allowEmpty);

        } else {
            primDesc = dnnl::convolution_forward::primitive_desc(
                convDesc, context->getEngine(), allowEmpty);
        }

        if (primDesc.get(allowEmpty) == nullptr)
            return false;

        // reorder data and weight
        auto srcMemory = userSrcMemory;
        if (primDesc.src_desc() != userSrcMemory.get_desc()) {
            srcMemory = dnnl::memory(primDesc.src_desc(), context->getEngine());

            prims.push_back(dnnl::reorder(userSrcMemory, srcMemory));
            primArgs.push_back(
                {{DNNL_ARG_FROM, userSrcMemory}, {DNNL_ARG_TO, srcMemory}});
        }

        auto wMemory = userWMemory;
        if (primDesc.weights_desc() != userWMemory.get_desc()) {
            wMemory =
                dnnl::memory(primDesc.weights_desc(), context->getEngine());

            prims.push_back(dnnl::reorder(userWMemory, wMemory));
            primArgs.push_back(
                {{DNNL_ARG_FROM, userWMemory}, {DNNL_ARG_TO, wMemory}});
        }

        // Create memory for output
        if (primDesc.dst_desc() == userDstMd) {
            auto output = dnnl::memory(primDesc.dst_desc(),
                                       context->getEngine(), dstData);

            // create convolution primitivee
            prims.push_back(dnnl::convolution_forward(primDesc));
            primArgs.push_back({{DNNL_ARG_SRC, srcMemory},
                                {DNNL_ARG_WEIGHTS, wMemory},
                                {DNNL_ARG_DST, output}});
        } else {
            auto dstMemory =
                dnnl::memory(primDesc.dst_desc(), context->getEngine());

            // create convolution primitivee
            prims.push_back(dnnl::convolution_forward(primDesc));
            primArgs.push_back({{DNNL_ARG_SRC, srcMemory},
                                {DNNL_ARG_WEIGHTS, wMemory},
                                {DNNL_ARG_DST, dstMemory}});

            auto output =
                dnnl::memory(userDstMd, context->getEngine(), dstData);
            prims.push_back(dnnl::reorder(dstMemory, output));
            primArgs.push_back(
                {{DNNL_ARG_FROM, dstMemory}, {DNNL_ARG_TO, output}});
        }
        return true;
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);
        auto record = as<ConvMklPerfRecordObj>(_record);

        std::vector<dnnl::primitive> prims;
        std::vector<std::unordered_map<int, dnnl::memory>> primArgs;
        IT_ASSERT(createPrimitives(op, record, context, true, prims, primArgs));

        IT_ASSERT(prims.size() == primArgs.size());
        for (size_t i = 0; i < prims.size(); ++i)
            prims.at(i).execute(context->getStream(), primArgs.at(i));
        context->getStream().wait();
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto record = make_ref<ConvMklPerfRecordObj>();
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        ConvMklPerfRecordObj ret;
        ret.time = std::numeric_limits<double>::max();
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);
        auto op = as<ConvObj>(_op);

        // Try every possible algorithm of convolution
        for (auto algo : {dnnl::algorithm::convolution_auto,
                          dnnl::algorithm::convolution_direct,
                          dnnl::algorithm::convolution_winograd}) {
            ConvMklPerfRecordObj record;
            record.algo = algo;

            std::vector<dnnl::primitive> prims;
            std::vector<std::unordered_map<int, dnnl::memory>> primArgs;
            if (!createPrimitives(op, make_ref<ConvMklPerfRecordObj>(record),
                                  context, true, prims, primArgs))
                continue;

            IT_ASSERT(prims.size() == primArgs.size());
            // does context->getStream() need to be attached to runtime, and
            // delete after each use?
            for (size_t i = 0; i < prims.size(); ++i)
                prims.at(i).execute(context->getStream(), primArgs.at(i));
            context->getStream().wait();

            record.time = timeit(
                [&]() {
                    for (size_t i = 0; i < prims.size(); ++i)
                        prims.at(i).execute(context->getStream(),
                                            primArgs.at(i));
                },
                [&]() { context->getStream().wait(); });

            // Update the tune result
            if (ret.time > record.time)
                ret = record;
        }

        IT_ASSERT(ret.time < std::numeric_limits<double>::max(), "No valid "
                                                                 "algorithm "
                                                                 "found");
        return make_ref<ConvMklPerfRecordObj>(ret);
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Conv, DataType::Float32, MklConv,
                "MklConv_CPU_float32");
} // namespace infini
