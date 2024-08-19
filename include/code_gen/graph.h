#ifndef GRAPH_H
#define GRAPH_H
#include "common.h"
#include "operator.h"
#include "tensor.h"
#include <memory>

namespace tpm {

class GraphBase {
  protected:
    OpVec ops;
    TensorVec tensors;
    TensorVec inputs;
    TensorVec outputs;

  public:
    virtual ~GraphBase();

    Tensor *tensor(const Dim &dims, Tensor::DataType dtype = Tensor::Float32);
    Tensor *tensor(const Dim &dims, const std::string &dtype);
    void addTensor(Tensor *tensor);
    TensorVec &getTensors();
    OpVec &getOperators();
    const OpVec &getOperators() const;
    TensorVec &getInputs();
    const TensorVec &getInputs() const;
    TensorVec &getOutputs();
    const TensorVec &getOutputs() const;

    void updateConnection();
    void removeOps(OpVec &ops);

    bool exportOnnx(const char *path);
    bool importOnnx(const char *net);
};

class Graph : public GraphBase {
  public:
    Graph() {}

    // conv op
    // bias is not part of the graph connections
    Operator *conv(Tensor *input, Tensor *weight, Tensor *output, int ph,
                   int pw, int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                   Tensor *bias = nullptr);
    Operator *conv(Tensor *input, Tensor *weight, int ph, int pw, int sh = 1,
                   int sw = 1, int dh = 1, int dw = 1, Tensor *bias = nullptr);
    Operator *conv(Tensor *input, Tensor *weight, Tensor *output,
                   ConvOp::PaddingMode pm, int sh = 1, int sw = 1, int dh = 1,
                   int dw = 1, Tensor *bias = nullptr);
    Operator *conv(Tensor *input, Tensor *weight, ConvOp::PaddingMode pm,
                   int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                   Tensor *bias = nullptr);
    // conv trans op
    // bias is not part of the graph connections
    Tensor *setConvTransInput(Tensor *input);
    Tensor *setConvTransWeight(Tensor *weight);
    Tensor *setConvTransOutput(Tensor *outputTrans, Tensor *output = nullptr);
    Operator *convTrans(Tensor *input, Tensor *weight, Tensor *output, int ph,
                        int pw, int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                        int oph = 0, int opw = 0, Tensor *bias = nullptr);
    Operator *convTrans(Tensor *input, Tensor *weight, int ph, int pw,
                        int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                        int oph = 0, int opw = 0, Tensor *bias = nullptr);
    Operator *convTrans(Tensor *input, Tensor *weight, Tensor *output,
                        ConvTransOp::PaddingMode pm, int sh = 1, int sw = 1,
                        int dh = 1, int dw = 1, int oph = 0, int opw = 0,
                        Tensor *bias = nullptr);
    Operator *convTrans(Tensor *input, Tensor *weight,
                        ConvTransOp::PaddingMode pm, int sh = 1, int sw = 1,
                        int dh = 1, int dw = 1, int oph = 0, int opw = 0,
                        Tensor *bias = nullptr);
    // matmul op
    // bias is not part of the graph connections
    Operator *matmul(Tensor *A, Tensor *B, Tensor *C, bool transA = false,
                     bool transB = false, Tensor *bias = nullptr);
    Operator *matmul(Tensor *A, Tensor *B, bool transA = false,
                     bool transB = false, Tensor *bias = nullptr);
    // g2bmm op
    // bias is not part of the graph connections
    Operator *g2bmm(Tensor *A, Tensor *B, Tensor *C, int width, int dilation,
                    Tensor *bias = nullptr);
    Operator *g2bmm(Tensor *A, Tensor *B, int width, int dilation,
                    Tensor *bias = nullptr);
    // gbmm-like op
    // bias is not part of the graph connections
    Operator *gbmml(Tensor *A, Tensor *B, Tensor *C, int dilation,
                    Tensor *bias = nullptr);
    Operator *gbmml(Tensor *A, Tensor *B, int dilation, Tensor *bias = nullptr);
    // pad op
    Operator *pad(Tensor *input, Tensor *output, const Dim &begin,
                  const Dim &end);
    Operator *pad(Tensor *input, const Dim &begin, const Dim &end);
    // slice op
    Operator *slice(Tensor *input, Tensor *output, const Dim &begin,
                    const Dim &end);
    Operator *slice(Tensor *input, const Dim &begin, const Dim &end);
    Operator *slice(Tensor *input, Tensor *output, Tensor *begin, Tensor *end);
    // concat op
    Operator *concat(const TensorVec &inputs, Tensor *output, int dim);
    Operator *concat(const TensorVec &inputs, int dim);
    // split op
    Operator *split(Tensor *input, const TensorVec &outputs, int dim, int num);
    Operator *split(Tensor *input, int dim, int num);
    Operator *split(Tensor *input, const TensorVec &outputs, int dim,
                    std::vector<int> sizes);
    Operator *split(Tensor *input, int dim, std::vector<int> sizes);
    // transpose op
    Operator *transpose(Tensor *input, Tensor *output, int split,
                        const Perm &after, int factor = 2);
    Operator *transpose(Tensor *input, int split, const Perm &after,
                        int factor = 2);
    // flatten (fake) op
    Operator *flatten(Tensor *input, Tensor *output, int axis);
    Operator *flatten(Tensor *input, int axis);
    // extend op
    Operator *extend(Tensor *input, Tensor *output, int dim, int num);
    Operator *extend(Tensor *input, int dim, int num);
    // batch norm op
    Operator *batchnorm(Tensor *input, Tensor *scale, Tensor *bias,
                        Tensor *mean, Tensor *var, Tensor *output,
                        float epsilon = 1e-05, float momentum = 0.9);
    Operator *batchnorm(Tensor *input, Tensor *scale, Tensor *bias,
                        Tensor *mean, Tensor *var, float epsilon = 1e-05,
                        float momentum = 0.9);
    // max pool op
    Operator *maxpool(Tensor *input, Tensor *output, int kh, int kw, int dh,
                      int dw, int ph, int pw, int sh, int sw);
    Operator *maxpool(Tensor *input, int kh, int kw, int dh, int dw, int ph,
                      int pw, int sh, int sw);
    // average pool op
    Operator *avgpool(Tensor *input, Tensor *output, int kh, int kw, int ph,
                      int pw, int sh, int sw);
    Operator *avgpool(Tensor *input, int kh, int kw, int ph, int pw, int sh,
                      int sw);
    // global average pool
    Operator *avgpool(Tensor *input, Tensor *output);
    // add op
    Operator *add(const TensorVec &inputs, Tensor *output);
    Operator *add(const TensorVec &inputs);
    // sub op
    Operator *sub(Tensor *input0, Tensor *input1, Tensor *output);
    Operator *sub(Tensor *input0, Tensor *input1);
    // mul op
    Operator *mul(const TensorVec &inputs, Tensor *output);
    Operator *mul(const TensorVec &inputs);
    // div op
    Operator *div(Tensor *input0, Tensor *input1, Tensor *output);
    Operator *div(Tensor *input0, Tensor *input1);
    // pow op
    Operator *pow(Tensor *input, Tensor *output, int pow);
    Operator *pow(Tensor *input, int pow);
    // gather op
    Operator *gather(Tensor *data, Tensor *indices, Tensor *output, int axis);
    Operator *gather(Tensor *data, Tensor *indices, int axis);
    // reduce mean op
    Operator *reduceMean(Tensor *input, Tensor *output, int axis);
    Operator *reduceMean(Tensor *input, int axis);
    // reshape op
    Operator *reshape(Tensor *input, Tensor *output);
    // identity op
    Operator *identity(Tensor *input, Tensor *output);
    Operator *identity(Tensor *input);
    // relu op
    Operator *relu(Tensor *input, Tensor *output);
    Operator *relu(Tensor *input);
    // sigmod op
    Operator *sigmoid(Tensor *input, Tensor *output);
    Operator *sigmoid(Tensor *input);
    // softmax op
    Operator *softmax(Tensor *input, Tensor *output, int axis);
    Operator *softmax(Tensor *input, int axis);
    Operator *tanh(Tensor *input, Tensor *output);
    Operator *tanh(Tensor *input);
    // membound op
    Operator *membound(const TensorVec &inputs, const TensorVec &outputs,
                       const std::vector<nnet::Tensor> &nnetInputs,
                       nnet::Expr expr, double exec_time,
                       std::string hint = {});
    // resize op
    Operator *resize(Tensor *input, Tensor *sizes, Tensor *output);
    Operator *resize(Tensor *input, Tensor *sizes);

    void setInputs(TensorVec inputs_);
    void setOutputs(TensorVec outputs_);
    bool importOnnx(const char *net);

    bool mutateInceptionHead();
};

class SubGraph : public GraphBase {
  private:
    int findTensor(Tensor *tensor, int ntensor = 0);
    uint64_t hash;

  public:
    SubGraph() {}
    SubGraph(OpVec oplist);
    SubGraph(const SubGraph &rhs) : SubGraph(rhs.ops) {}
    void cleanConnection();
    bool resetOps(OpVec oplist, size_t ntensor = 0);
    /**
     * Compute one point of one output
     * @return : pair(success?, value)
     */
    const std::pair<bool, VType> compute(const Dim &point, size_t outputId = 0,
                                         bool getAllPos = false) const;
    uint64_t getHash();
    // level 1 considers inputs and outputs while level 2 not
    uint64_t getCacheIdx(int level);
    int print();
    int printBrief();
    int getComputeOps(std::vector<Operator *> &ops);
    // replace input and output tensor.
    int reset(std::vector<Tensor *> &input, std::vector<Tensor *> &output);
    // split graph to master and slave.
    int split(std::shared_ptr<SubGraph> &master,
              std::shared_ptr<SubGraph> &slave, std::vector<Operator *> &ops);
    // merge graph and slave to master, slave will be freed after merge.
    int merge(std::shared_ptr<SubGraph> &master,
              std::shared_ptr<SubGraph> &slave);

    bool compute();
    bool verification(SubGraph *mutant_graph, bool isFullComputing);
};

} // end of namespace tpm

#endif // GRAPH_H
