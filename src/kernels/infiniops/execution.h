#pragma once
#include "core/common.h"
#include <cmath>
#include <functional>
#include <infini/ops.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

namespace infini::infiniops {
using TensorView = ::infini::ops::Tensor;
using DataType = ::infini::rt::DataType;
using Handle = ::infini::ops::Handle;
using Value =
    std::variant<std::monostate, TensorView, std::vector<TensorView>, int64_t,
                 double, bool, std::string, std::vector<int64_t>, DataType>;
using Values = std::vector<Value>;
using NamedValues = std::map<std::string, Value>;
using WorkspaceAllocator = std::function<std::shared_ptr<void>(size_t)>;
struct Step {
    std::function<void(const Handle &)> run;
    bool captureSafe;
    std::string name;
    std::string provider = {};
    size_t workspaceBytes = 0;
    uint64_t nodeId = 0;
};
struct Plan {
    // Steps own prepared operators borrowing these temporary tensor views.
    // Declare storage first so it outlives operator destruction.
    std::vector<std::shared_ptr<void>> resources;
    std::vector<Step> steps;
    void run(const Handle &handle) const;
};

template <typename T> struct Optional : std::false_type {};
template <typename T> struct Optional<std::optional<T>> : std::true_type {
    using Type = T;
};
template <typename T> T getArgument(const Value &value) {
    if constexpr (Optional<T>::value) {
        if (std::holds_alternative<std::monostate>(value))
            return std::nullopt;
        return getArgument<typename Optional<T>::Type>(value);
    } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        const auto number = std::get<int64_t>(value);
        IT_ASSERT(
            static_cast<long double>(number) >=
                    static_cast<long double>(
                        std::numeric_limits<T>::lowest()) &&
                static_cast<long double>(number) <=
                    static_cast<long double>(std::numeric_limits<T>::max()),
            "Integer attribute out of range");
        return static_cast<T>(number);
    } else if constexpr (std::is_floating_point_v<T>) {
        const auto number = std::holds_alternative<int64_t>(value)
                                ? static_cast<double>(std::get<int64_t>(value))
                                : std::get<double>(value);
        IT_ASSERT((!std::isfinite(number) ||
                   std::abs(number) <= std::numeric_limits<T>::max()),
                  "Floating attribute out of range");
        return static_cast<T>(number);
    } else {
        return std::get<T>(value);
    }
}
template <typename T> Value argument(const T &value) {
    if constexpr (Optional<T>::value) {
        return value ? argument(*value) : Value{std::monostate{}};
    } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        IT_ASSERT(
            static_cast<long double>(value) <=
                static_cast<long double>(std::numeric_limits<int64_t>::max()),
            "Integer attribute out of range");
        return int64_t(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return double(value);
    } else {
        return value;
    }
}

template <typename Key, typename... Args>
Step prepareTyped(const WorkspaceAllocator &allocate, const Args &...args) {
    auto config = ::infini::ops::Config::Automatic();
    std::shared_ptr<::infini::ops::Operator<Key>> op =
        Key::Make(config, args...);
    const auto bytes = op->workspace_size_in_bytes();
    auto workspace = bytes ? allocate(bytes) : std::shared_ptr<void>{};
    IT_ASSERT(!bytes || workspace, "Missing operator workspace");
    const auto safe = op->capture_safe();
    return {[op, workspace, bytes, args...](const Handle &handle) {
                auto invocation = handle.Clone();
                if (bytes) {
                    invocation->set_workspace(workspace.get());
                    invocation->set_workspace_size_in_bytes(bytes);
                }
                (*op)(*invocation, args...);
            },
            safe,
            {},
            typeid(*op).name(),
            bytes};
}

struct Parameter {
    std::string name;
    std::string type;
};
class Registry {
    using Factory =
        std::function<Step(const Values &, const WorkspaceAllocator &)>;
    struct Overload {
        std::vector<Parameter> parameters;
        Factory factory;
    };
    std::map<std::string, std::vector<Overload>> operations;

  public:
    static Registry &instance();
    void add(std::string name, std::vector<Parameter> parameters,
             Factory factory);
    Step prepare(const std::string &name, const Values &values,
                 const WorkspaceAllocator &allocate) const;
    Step prepare(const std::string &name, const NamedValues &values,
                 const WorkspaceAllocator &allocate) const;
    bool simple(const std::string &name, size_t inputs, size_t outputs) const;
    std::string schemas() const;
};
void registerGeneratedBindings(Registry &registry);
class Builder {
    WorkspaceAllocator allocate;

  public:
    Plan plan;
    explicit Builder(WorkspaceAllocator allocate)
        : allocate(std::move(allocate)) {}
    template <typename... Args>
    void emit(const std::string &name, const Args &...args) {
        plan.steps.push_back(Registry::instance().prepare(
            name, Values{argument(args)...}, allocate));
    }
    void emit(const std::string &name, const NamedValues &values) {
        plan.steps.push_back(
            Registry::instance().prepare(name, values, allocate));
    }
    std::shared_ptr<void> temporary(size_t bytes) {
        auto data = allocate(bytes);
        plan.resources.push_back(data);
        return data;
    }
};
} // namespace infini::infiniops
