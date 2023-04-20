#pragma once
#include "nlohmann/json_fwd.hpp"
#include "nnet/visitor.h"
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
    string visit_(const Func &c) override;
    string dispatchRoutine(const Routine &c);

    Expr buildExprTree(string key);
    Routine buildRoutine(string key);

  public:
    Serializer(int _verobse = 0);
    virtual ~Serializer();

    /**
     * @brief Serialize the given expression to string
     *
     * @param expr The expression to be serialized
     * @param msg Message of derivation
     * @param inputs membound operator attributes
     * @param exec_time membound operator attributes
     * @param hint membound operator attributes
     * @return bool Whether the serialization succeed
     */
    std::optional<std::string> toString(Expr const &expr,
                                        const string &msg = "",
                                        vector<Tensor> inputs = {},
                                        double exec_time = -1e9,
                                        string hint = "");

    /**
     * @brief Serialize the given expression to json file
     *
     * @param expr The expression to be serialized
     * @param filePath The path of json file to be output
     * @param msg Message of derivation
     * @param inputs membound operator attributes
     * @param exec_time membound operator attributes
     * @param hint membound operator attributes
     * @return bool Whether the serialization succeed
     */
    bool toFile(const Expr &expr, const string &filePath,
                const string &msg = "", vector<Tensor> inputs = {},
                double exec_time = -1e9, string hint = "");

    /**
     * @brief Deserialize the given json file to expression
     *
     * @param text The text of the expr to be deserialized
     * @return Expression deserialized from the given json file
     */
    Expr fromString(const string &text);

    /**
     * @brief Deserialize the given json file to expression
     *
     * @param filePath The path to file to be deserialized
     * @return Expression deserialized from the given json file
     */
    Expr fromFile(const string &filePath);

    tuple<Expr, vector<Tensor>, double, string>
    deserializeAsMemobundOp(const string &filePath);

    // FIXME: the order of elements in tuple is not consistent with memboundObj
    // constructor
    tuple<Expr, vector<Tensor>, double, string>
    membundOpFromString(const string &data);
};

} // namespace nnet
