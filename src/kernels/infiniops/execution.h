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
using MemoryAllocator = std::function<std::shared_ptr<void>(size_t)>;
struct Workspace {
    size_t bytes = 0;
    std::shared_ptr<void> data;
};
using WorkspaceAllocator = std::function<std::shared_ptr<Workspace>(size_t)>;
struct Step {
    std::function<void(const Handle &)> run;
    bool captureSafe;
    std::string name;
};
struct Plan {
    // Steps own prepared operators borrowing these temporary tensor views.
    // Declare storage first so it outlives operator destruction.
    std::vector<std::shared_ptr<void>> resources;
    size_t temporaryBytes = 0;
    std::shared_ptr<void> temporaryStorage;
    std::shared_ptr<Workspace> workspace;
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
    auto workspace = bytes ? allocate(bytes) : std::shared_ptr<Workspace>{};
    IT_ASSERT(!bytes || workspace, "Missing operator workspace");
    const auto safe = op->capture_safe();
    return {[op, workspace, bytes, args...](const Handle &handle) {
                auto invocation = handle.Clone();
                if (bytes) {
                    IT_ASSERT(workspace->data && workspace->bytes >= bytes,
                              "Unallocated operator workspace");
                    invocation->set_workspace(workspace->data.get());
                    invocation->set_workspace_size_in_bytes(bytes);
                }
                (*op)(*invocation, args...);
            },
            safe,
            {}};
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
    MemoryAllocator allocate;
    bool planning;
    size_t temporaryCursor = 0, peak = 0;
    static std::shared_ptr<void> placeholder() {
        static char unused;
        return {&unused, [](void *) {}};
    }
    WorkspaceAllocator workspaceRequests() {
        return [this](size_t bytes) {
            plan.workspace->bytes = std::max(plan.workspace->bytes, bytes);
            return plan.workspace;
        };
    }

  public:
    Plan plan;
    explicit Builder(MemoryAllocator allocate, size_t temporaryBytes = 0,
                     bool planning = false)
        : allocate(std::move(allocate)), planning(planning) {
        plan.workspace = std::make_shared<Workspace>();
        plan.temporaryBytes = temporaryBytes;
        if (!planning && temporaryBytes)
            plan.temporaryStorage = this->allocate(temporaryBytes);
    }
    // Helpers live through one IR node, including all its lowered steps.
    // Nodes and their kernels execute in order on the same runtime stream.
    void beginNode() { temporaryCursor = 0; }
    size_t temporaryPeak() const {
        // Round the final helper as well: capacity then equals the sum of
        // aligned sizes, independent of C++ argument evaluation order.
        IT_ASSERT(peak <= std::numeric_limits<size_t>::max() - 255,
                  "Temporary layout overflow");
        return (peak + 255) / 256 * 256;
    }
    void finalize() {
        if (!planning && plan.workspace->bytes)
            plan.workspace->data = allocate(plan.workspace->bytes);
    }
    template <typename... Args>
    void emit(const std::string &name, const Args &...args) {
        emit(name, Values{argument(args)...});
    }
    void emit(const std::string &name, const Values &values) {
        if (!planning)
            plan.steps.push_back(Registry::instance().prepare(
                name, values, workspaceRequests()));
    }
    void emit(const std::string &name, const NamedValues &values) {
        if (!planning)
            plan.steps.push_back(Registry::instance().prepare(
                name, values, workspaceRequests()));
    }
    std::shared_ptr<void> temporary(size_t bytes) {
        constexpr size_t alignment = 256;
        IT_ASSERT(temporaryCursor <=
                      std::numeric_limits<size_t>::max() - (alignment - 1),
                  "Temporary layout overflow");
        const auto offset =
            (temporaryCursor + alignment - 1) / alignment * alignment;
        IT_ASSERT(bytes <= std::numeric_limits<size_t>::max() - offset,
                  "Temporary layout overflow");
        temporaryCursor = offset + bytes;
        peak = std::max(peak, temporaryCursor);
        if (planning)
            return placeholder();
        IT_ASSERT(temporaryCursor <= plan.temporaryBytes,
                  "Temporary layout changed during preparation");
        if (!bytes)
            return {};
        return {plan.temporaryStorage,
                static_cast<uint8_t *>(plan.temporaryStorage.get()) + offset};
    }
    // Read-only values must not alias the scratch area overwritten by nodes.
    std::shared_ptr<void>
    constant(size_t bytes, const std::function<void(void *)> &initialize) {
        if (planning)
            return placeholder();
        auto data = allocate(bytes);
        initialize(data.get());
        plan.resources.push_back(data);
        return data;
    }
};
} // namespace infini::infiniops
