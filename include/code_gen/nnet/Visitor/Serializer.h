#pragma once
#include "code_gen/nnet/visitor.h"
#include "nlohmann/json_fwd.hpp"
#include <memory>

namespace nnet {

class Serializer : public Functor<string()> {
    using json = nlohmann::ordered_json;

  private:
    static constexpr int VERSION{1};
    std::unique_ptr<json> jPtr;
    json &j;
    static int id;

    string visit_(const Constant &c) override;
    string visit_(const BinaryOp &c) override;
    string visit_(const RangeOp &c) override;
    string visit_(const Subscript &c) override;
    string visit_(const Var &c) override;
    string visit_(const Tensor &c) override;
    string dispatchRoutine(const Routine &c);

    Expr buildExprTree(string key);
    Routine buildRoutine(string key);

  public:
    Serializer(int _verobse = 0);
    virtual ~Serializer();

    /**
     * @brief Serialize the given expression to json file
     *
     * @param expr The expression to be serialized
     * @param filePath The path of json file to be output
     * @param msg Message of derivation
     * @return bool Whether the serialization succeed
     */
    bool serialize(const Expr &expr, const string &filePath,
                   const string &msg = "");

    /**
     * @brief Deserialize the given json file to expression
     *
     * @param filePath The path to file to be deserialized
     * @return Expression deserialized from the given json file
     */
    Expr deserialize(const string &filePath);
};

} // namespace nnet