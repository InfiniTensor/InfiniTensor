#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/conv.h"

namespace infini {
struct ConvTransposeMklPerfRecordObj : public PerfRecordObj {
    dnnl::algorithm algo = dnnl::algorithm::deconvolution_direct;
    void to_json(json &j) override {
        j["type"] = 1;
        j["data"] = std::make_tuple(enum_to_underlying(algo), time);
    }
    static PerfRecord from_json(const json &j) {
        ConvTransposeMklPerfRecordObj tmp;
        auto [Algo, Time] = j["data"].get<tuple<int, double>>();
        tmp.algo = (dnnl::algorithm)Algo;
        tmp.time = Time;
        return make_ref<ConvTransposeMklPerfRecordObj>(tmp);
    }
};

using ConvTransposeMklPerfRecord = Ref<ConvTransposeMklPerfRecordObj>;
class MklConvTranspose : public Kernel {
  private:
    bool createPrimitives(
        const Ref<ConvTransposed2dObj> &op,
        const ConvTransposeMklPerfRecord &record, const MklRuntimeObj *context,
        bool allowEmpty, std::vector<dnnl::primitive> &prims,
        std::vector<std::unordered_map<int, dnnl::memory>> &primArgs) const {
        auto srcData = op->getInputs(0)->getRawDataPtr<float *>();
        auto wData = op->getInputs(1)->getRawDataPtr<float *>();
        // FIXME: iohw2iohwData
        auto dstData = op->getOutput(0)->getRawDataPtr<float *>();

        auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const int cpg = op->getChannelPerGroup();
        if (cpg != c)
            IT_TODO_HALT();

        auto oDims = op->getOutput(0)->getDims();
        int oH = oDims[oDims.size() - 2];
        int oW = oDims[oDims.size() - 1];

        //  create user memory that describes data layout in the buffers
        auto userSrcMd =
            dnnl::memory::desc({n, f, h, w}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nchw);
        auto userSrcMemory =
            dnnl::memory(userSrcMd, context->getEngine(), srcData);

        // DNNL deconvolution expects the logical order of weights (parameters)
        // dimensions to be in order {o, i, h, w}. So need to reorder wData.
        // TODO: to make reorder happen only once when inference (because
        // weights are fixed).
        // TODO: Fix by whj, change memory format tag from oihw to iohw to
        // remove extra transpose. Correctness to be confirmed.
        auto userWMd =
            dnnl::memory::desc({cpg, f, r, s}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::iohw);

        auto userWMemory = dnnl::memory(userWMd, context->getEngine(), wData);

        auto userDstMd =
            dnnl::memory::desc({n, c, oH, oW}, dnnl::memory::data_type::f32,
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
            dnnl::memory::desc({n, f, h, w}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);
        auto wMd =
            dnnl::memory::desc({cpg, f, r, s}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);
        auto dstMd =
            dnnl::memory::desc({n, c, oH, oW}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);

        // create convolution descriptor
        dnnl::memory::dims strides = {sh, sw};
        dnnl::memory::dims pads = {ph, pw};
        dnnl::memory::dims dilations = {dh - 1, dw - 1};
        auto deconvDesc = dnnl::deconvolution_forward::desc(
            dnnl::prop_kind::forward_inference, record->algo, srcMd, wMd, dstMd,
            strides, dilations, pads, pads);

        dnnl::deconvolution_forward::primitive_desc primDesc;
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

            primDesc = dnnl::deconvolution_forward::primitive_desc(
                deconvDesc, attr, context->getEngine(), allowEmpty);

        } else {
            primDesc = dnnl::deconvolution_forward::primitive_desc(
                deconvDesc, context->getEngine(), allowEmpty);
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

        if (primDesc.dst_desc() == userDstMd) {
            // Create memory for output
            auto dstMemory = dnnl::memory(primDesc.dst_desc(),
                                          context->getEngine(), dstData);

            // create convolution primitivee
            prims.push_back(dnnl::deconvolution_forward(primDesc));
            primArgs.push_back({{DNNL_ARG_SRC, srcMemory},
                                {DNNL_ARG_WEIGHTS, wMemory},
                                {DNNL_ARG_DST, dstMemory}});
        } else {
            auto dstMemory =
                dnnl::memory(primDesc.dst_desc(), context->getEngine());

            // create convolution primitivee
            prims.push_back(dnnl::deconvolution_forward(primDesc));
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
                 const RuntimeObj *_context) const {
        auto op = as<ConvTransposed2dObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);
        auto record = as<ConvTransposeMklPerfRecordObj>(_record);

        dnnl::stream stream(context->getEngine());
        std::vector<dnnl::primitive> prims;
        std::vector<std::unordered_map<int, dnnl::memory>> primArgs;
        IT_ASSERT(createPrimitives(op, record, context, true, prims, primArgs));

        IT_ASSERT(prims.size() == primArgs.size());
        for (size_t i = 0; i < prims.size(); ++i)
            prims.at(i).execute(stream, primArgs.at(i));
        stream.wait();
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto record = make_ref<ConvTransposeMklPerfRecordObj>();
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        ConvTransposeMklPerfRecordObj ret;
        ret.time = std::numeric_limits<double>::max();
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);
        auto op = as<ConvTransposed2dObj>(_op);

        // Try every possible algorithm of convolution
        for (auto algo : {dnnl::algorithm::deconvolution_direct,
                          dnnl::algorithm::deconvolution_winograd}) {
            ConvTransposeMklPerfRecordObj record;
            record.algo = algo;

            std::vector<dnnl::primitive> prims;
            std::vector<std::unordered_map<int, dnnl::memory>> primArgs;
            if (!createPrimitives(
                    op, make_ref<ConvTransposeMklPerfRecordObj>(record),
                    context, true, prims, primArgs))
                continue;

            IT_ASSERT(prims.size() == primArgs.size());
            dnnl::stream stream(context->getEngine());
            for (size_t i = 0; i < prims.size(); ++i)
                prims.at(i).execute(stream, primArgs.at(i));
            stream.wait();

            record.time = timeit(
                [&]() {
                    for (size_t i = 0; i < prims.size(); ++i)
                        prims.at(i).execute(stream, primArgs.at(i));
                },
                [&]() { stream.wait(); });

            // Update the tune result
            if (ret.time > record.time)
                ret = record;
        }

        IT_ASSERT(ret.time < std::numeric_limits<double>::max(), "No valid "
                                                                 "algorithm "
                                                                 "found");
        return make_ref<ConvTransposeMklPerfRecordObj>(ret);
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::ConvTrans, DataType::Float32,
                MklConvTranspose, "MklConvTrans_CPU_float32");

} // namespace infini
