#include "execution.h"
#include "operators/rms_norm.h"
#include "tensor_view.h"
#include <cctype>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace infini::infiniops {
namespace {
Value normalize(const std::string &type, const Value &value) {
    auto base = type;
    if (base.rfind("std::optional<", 0) == 0)
        base = base.substr(14, base.size() - 15);
    if (base == "std::vector<Tensor>" &&
        std::holds_alternative<TensorView>(value))
        return std::vector<TensorView>{std::get<TensorView>(value)};
    if (base == "DataType" && !std::holds_alternative<std::monostate>(value)) {
        if (std::holds_alternative<int64_t>(value)) {
            const auto index = std::get<int64_t>(value);
            IT_ASSERT(index > 0 && index < static_cast<int64_t>(std::size(
                                               ::infini::DataType::names)),
                      "Invalid ONNX dtype code");
            return toInfiniOpsDataType(
                ::infini::DataType(static_cast<int>(index)));
        }
        if (std::holds_alternative<std::string>(value)) {
            auto text = std::get<std::string>(value);
            std::transform(text.begin(), text.end(), text.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            for (size_t index = 1; index < std::size(::infini::DataType::names);
                 ++index) {
                if (::infini::DataType::cpuType[index] < 0)
                    continue;
                auto dtype = toInfiniOpsDataType(
                    ::infini::DataType(static_cast<int>(index)));
                auto name =
                    std::string(::infini::rt::kDataTypeToDesc.at(dtype));
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (name == text)
                    return toInfiniOpsDataType(
                        ::infini::DataType(static_cast<int>(index)));
            }
            IT_ASSERT(false, "Unknown dtype name: " + text);
        }
    }
    return value;
}
bool accepts(const std::string &type, const Value &value) {
    if (type.rfind("std::optional<", 0) == 0) {
        return std::holds_alternative<std::monostate>(value) ||
               accepts(type.substr(14, type.size() - 15), value);
    }
    if (type == "Tensor")
        return std::holds_alternative<TensorView>(value);
    if (type == "std::vector<Tensor>")
        return std::holds_alternative<std::vector<TensorView>>(value);
    if (type == "bool")
        return std::holds_alternative<bool>(value);
    if (type == "std::string")
        return std::holds_alternative<std::string>(value);
    if (type == "std::vector<int64_t>")
        return std::holds_alternative<std::vector<int64_t>>(value);
    if (type == "DataType")
        return std::holds_alternative<DataType>(value);
    if (type == "float" || type == "double")
        return std::holds_alternative<double>(value) ||
               std::holds_alternative<int64_t>(value);
    return std::holds_alternative<int64_t>(value);
}
} // namespace
void Plan::run(const Handle &handle) const {
    for (const auto &step : steps)
        step.run(handle);
}
Registry &Registry::instance() {
    static Registry registry = [] {
        Registry value;
        registerGeneratedBindings(value);
        return value;
    }();
    return registry;
}
void Registry::add(std::string name, std::vector<Parameter> parameters,
                   Factory factory) {
    operations[std::move(name)].push_back(
        {std::move(parameters), std::move(factory)});
}
Step Registry::prepare(const std::string &name, const Values &values,
                       const WorkspaceAllocator &allocate) const {
    const auto found = operations.find(name);
    IT_ASSERT(found != operations.end(),
              "InfiniOps SDK does not export operator '" + name + "'");
    for (const auto &overload : found->second) {
        if (values.size() != overload.parameters.size())
            continue;
        bool matches = true;
        for (size_t i = 0; i < values.size(); ++i)
            matches &= accepts(overload.parameters[i].type, values[i]);
        if (matches) {
            try {
                // The pinned SDK does not check the weight bounds itself.
                // Keep this shared by legacy IR, generic nodes and NativeCpu.
                if (name == "RmsNorm") {
                    const auto &input = std::get<TensorView>(values.at(0));
                    const auto &weight = std::get<TensorView>(values.at(1));
                    const auto &out = std::get<TensorView>(values.back());
                    validateRmsNormShape(input.shape(), weight.numel());
                    IT_ASSERT(input.ndim() == 2 || input.ndim() == 3,
                              "The pinned RmsNorm SDK requires rank 2 or 3");
                    IT_ASSERT(input.shape() == out.shape(),
                              "RMSNorm output shape must match input");
                    IT_ASSERT(input.dtype() == weight.dtype() &&
                                  input.dtype() == out.dtype(),
                              "RMSNorm tensor dtypes must match");
                }
                auto step = overload.factory(values, allocate);
                step.name = name;
                return step;
            } catch (const std::exception &error) {
                throw std::runtime_error("InfiniOps prepare '" + name +
                                         "': " + error.what());
            }
        }
    }
    IT_ASSERT(false,
              "InfiniOps argument signature mismatch for '" + name + "'");
    return {};
}
Step Registry::prepare(const std::string &name, const NamedValues &values,
                       const WorkspaceAllocator &allocate) const {
    const auto found = operations.find(name);
    IT_ASSERT(found != operations.end(),
              "InfiniOps SDK does not export operator '" + name + "'");
    for (const auto &overload : found->second) {
        Values ordered;
        size_t consumed = 0;
        bool matches = true;
        for (const auto &param : overload.parameters) {
            const auto value = values.find(param.name);
            if (value == values.end()) {
                if (param.type.rfind("std::optional<", 0) != 0) {
                    matches = false;
                    break;
                }
                ordered.emplace_back(std::monostate{});
            } else {
                auto normalized = normalize(param.type, value->second);
                matches &= accepts(param.type, normalized);
                ordered.push_back(std::move(normalized));
                ++consumed;
            }
        }
        if (matches && consumed == values.size())
            return prepare(name, ordered, allocate);
    }
    IT_ASSERT(false,
              "InfiniOps named argument signature mismatch for '" + name + "'");
    return {};
}
bool Registry::simple(const std::string &name, size_t inputs,
                      size_t outputs) const {
    const auto found = operations.find(name);
    if (found == operations.end() || outputs != 1)
        return false;
    for (const auto &overload : found->second) {
        if (overload.parameters.size() != inputs + 1 ||
            overload.parameters.back().name != "out")
            continue;
        bool match = true;
        for (const auto &p : overload.parameters)
            match &= p.type == "Tensor";
        if (match)
            return true;
    }
    return false;
}
std::string Registry::schemas() const {
    auto result = nlohmann::json::object();
    for (const auto &[name, overloads] : operations) {
        for (const auto &overload : overloads) {
            auto params = nlohmann::json::array();
            for (const auto &p : overload.parameters)
                params.push_back({{"name", p.name}, {"type", p.type}});
            result[name].push_back(params);
        }
    }
    return result.dump();
}
} // namespace infini::infiniops
