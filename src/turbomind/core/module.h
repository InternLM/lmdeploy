// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TURBOMIND_CORE_MODULE_H
#define TURBOMIND_CORE_MODULE_H

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

// ======================================================================
// X-macro config field infrastructure
// ======================================================================

#define TM_MEMBER(Type, name, ...) Type name{__VA_ARGS__};
#define TM_PTR(Type, name, ...) visitor(#name, &Config::name);
#define TM_FOR_EACH(ClassName, field_list)                                                                             \
    template<typename Visitor>                                                                                         \
    static void for_each(Visitor&& visitor)                                                                            \
    {                                                                                                                  \
        using Config = ClassName;                                                                                      \
        field_list(TM_PTR)                                                                                             \
    }

// ======================================================================
// ModuleConfig — plain base for typed config structs
// ======================================================================

struct ModuleConfig {
    std::string_view module_type;
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    template<typename Visitor>
    static void for_each(Visitor&&)
    {
    }
};

// ======================================================================
// X-macro expansion macros for Module-derived classes
//
// Usage in a derived class header:
//
//   #define MY_CHILDREN(X) \
//       X(LinearWeight, w1) \
//       X(NormWeight,   norm)
//
//   #define MY_PARAMS(X) \
//       X(weight) \
//       X(bias)
//
//   class MyWeight: public Module {
//   public:
//       MY_CHILDREN(TM_CHILD_MEMBER)
//       MY_PARAMS(TM_PARAM_MEMBER)
//
//       // Optional: override virtuals using the CASE macros
//       Module* add_child(std::string name, std::unique_ptr<Module> child) override;
//       Module* child(const std::string& name) const override;
//       Param param(const std::string& name) override;
//       void for_each_child(std::function<void(const char*, Module*)> visitor) const override;
//       void for_each_param(std::function<void(const char*, Tensor&)> visitor) override;
//   };
//
//   // In the .cc file:
//   Module* MyWeight::add_child(std::string name, std::unique_ptr<Module> child) {
//       MY_CHILDREN(TM_ADD_CHILD_CASE)
//       return nullptr;
//   }
//   // ... etc.
// ======================================================================

/// Declares a unique_ptr<Type> member named `name`.
#define TM_CHILD_MEMBER(Type, name) std::unique_ptr<Type> name;

/// Declares a Tensor member named `name`.
#define TM_PARAM_MEMBER(name) core::Tensor name{};

/// Fragment for add_child() override body: matches name and stores child.
/// Assumes member `std::unique_ptr<Type> name` and local `std::string name_str`.
#define TM_ADD_CHILD_CASE(Type, name)                                                                                  \
    if (name_str == #name) {                                                                                           \
        TM_CHECK_EQ(child->type(), Type().type());                                                                     \
        name.reset(static_cast<Type*>(child.release()));                                                               \
        attach_child_(name.get(), this, std::move(name_str));                                                          \
        return name.get();                                                                                             \
    }

/// Fragment for child() override body: matches name and returns pointer.
#define TM_CHILD_CASE(Type, name)                                                                                      \
    if (name_str == #name) {                                                                                           \
        return name.get();                                                                                             \
    }

/// Fragment for param() override body: matches name and returns Param handle.
#define TM_PARAM_CASE(name)                                                                                            \
    if (name_str == #name) {                                                                                           \
        return core::Param{&name};                                                                                     \
    }

/// Fragment for for_each_child() override body: visits child.
#define TM_VISIT_CHILD(Type, name) visitor(#name, name.get());

/// Fragment for for_each_param() override body: visits param.
#define TM_VISIT_PARAM(name) visitor(#name, name);

/// Declares data members (children + params) and virtual method overrides.
/// Used in the public section of a derived class.
#define TM_MODULE_DECLARE(Class, ChildrenX, ParamsX)                                                                   \
    ChildrenX(TM_CHILD_MEMBER) ParamsX(TM_PARAM_MEMBER) core::Module* add_child(                                       \
        std::string name, std::unique_ptr<Module> child) override;                                                     \
    core::Module* child(const std::string& name) const override;                                                       \
    core::Param   param(const std::string& name) override;                                                             \
    void          for_each_child(std::function<void(const char*, Module*)> visitor) const override;                    \
    void          for_each_param(std::function<void(const char*, core::Tensor&)> visitor) override;

/// Defines all X-macro generated method bodies for a derived module class.
/// Used in the .cc file.  ChildrenX/ParamsX may be empty macros.
#define TM_MODULE_METHODS(Class, ChildrenX, ParamsX)                                                                   \
    core::Module* Class::add_child(std::string name, std::unique_ptr<core::Module> child)                              \
    {                                                                                                                  \
        std::string name_str = std::move(name);                                                                        \
        ChildrenX(TM_ADD_CHILD_CASE) return nullptr;                                                                   \
    }                                                                                                                  \
    core::Module* Class::child(const std::string& name_str) const                                                      \
    {                                                                                                                  \
        ChildrenX(TM_CHILD_CASE) return nullptr;                                                                       \
    }                                                                                                                  \
    core::Param Class::param(const std::string& name_str)                                                              \
    {                                                                                                                  \
        ParamsX(TM_PARAM_CASE) return {};                                                                              \
    }                                                                                                                  \
    void Class::for_each_child(std::function<void(const char*, core::Module*)> visitor) const                          \
    {                                                                                                                  \
        ChildrenX(TM_VISIT_CHILD)                                                                                      \
    }                                                                                                                  \
    void Class::for_each_param(std::function<void(const char*, core::Tensor&)> visitor)                                \
    {                                                                                                                  \
        ParamsX(TM_VISIT_PARAM)                                                                                        \
    }

// ======================================================================
// Param — lightweight handle to a Module parameter slot
// ======================================================================

/// Lightweight handle to a Tensor slot within a Module.
/// Returned by Module::param(name). Used for per-param allocation.
class Param {
    Tensor* slot_;

public:
    Param(Tensor* slot = nullptr): slot_(slot) {}

    /// Allocate the tensor with explicit shape/dtype. Returns the tensor for data copy.
    Tensor alloc(const std::vector<size_t>& shape, DataType dtype)
    {
        TM_CHECK(slot_ != nullptr);
        auto layout = Layout{std::vector<ssize_t>(shape.begin(), shape.end())};
        *slot_      = Tensor{std::move(layout), dtype, kDEVICE};
        return *slot_;
    }

    /// Get current tensor (empty if not yet allocated).
    Tensor get() const
    {
        return slot_ ? *slot_ : Tensor{};
    }

    explicit operator bool() const
    {
        return slot_ && static_cast<bool>(*slot_);
    }
};

// ======================================================================
// Module — type-erased hierarchical module with virtual lifecycle
// ======================================================================

/// Type-erased hierarchical module with virtual lifecycle.
///
/// The module tree is built explicitly via ``create_child()`` from the Python
/// loading pipeline. Children are looked up by name; no lazy creation.
///   - ``prepare()`` runs post-load processing (format conversion, fusion).
///   - ``verify()`` walks the tree and collects uninitialized params/modules.
///
/// Derived classes use X-macro hooks (TM_CHILD_MEMBER, TM_PARAM_MEMBER, etc.)
/// to declare children and parameters as direct members, overriding the
/// virtual lookup methods to match by name.
class Module {
    friend class ModuleList;

public:
    virtual ~Module();

    Module();

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&)                 = delete;
    Module& operator=(Module&&) = delete;

    // ----- Type info -----

    /// Returns a static string identifying the module type (e.g., "LinearWeight", "NormWeight").
    virtual const char* type() const;

    // ----- Hierarchy (virtual, overridden by derived classes) -----

    /// Owns child; registers it under the given local name.
    /// Returns raw pointer to the added child, or nullptr if name not recognized.
    /// Default: returns nullptr.
    virtual Module* add_child(std::string name, std::unique_ptr<Module> child);

    /// Find a direct child by name. Default: returns nullptr.
    virtual Module* child(const std::string& name) const;

    /// Iterate over all children. Default: no-op.
    virtual void for_each_child(std::function<void(const char*, Module*)> visitor) const;

    // ----- Parameters (virtual, overridden by derived classes) -----

    /// Find a parameter by name within this module. Default: returns empty Param.
    virtual Param param(const std::string& name);

    /// Iterate over all parameters. Default: no-op.
    virtual void for_each_param(std::function<void(const char*, Tensor&)> visitor);

    // ----- Lifecycle (virtual, default = recurse / no-op) -----

    /// Post-load processing: weight format conversion, fusion.
    /// Default recurses into children via for_each_child.
    virtual void prepare();

    // ----- Registry-driven child creation -----

    /// Create a standalone module using the type registry (no parent binding).
    static std::unique_ptr<Module> create(const ModuleConfig& config);

    /// Create a child module using the type registry and attach it.
    /// Uses config.module_type to look up the factory.
    /// Returns pointer to the created child, or nullptr on failure.
    Module* create_child(const std::string& name, const ModuleConfig& config = {});

    /// Typed child accessor. Aborts if child not found.
    template<typename T>
    T* get(const std::string& name) const
    {
        auto* c = child(name);
        TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
        return static_cast<T*>(c);
    }

    /// Find a child by single segment name. Aborts on null.
    Module* get(const std::string& segment);

    // ----- Verification -----

    /// Walk subtree, collect paths of uninitialized params/modules into ``missing``.
    /// Composite modules override to also check required children exist.
    /// Returns true if everything is OK.
    virtual bool verify(std::vector<std::string>& missing);

    // ----- Utilities -----

    /// Build the fully-qualified path by walking up the parent chain.
    std::string full_path() const;

    /// Access the parent module (nullptr for root).
    Module* parent() const noexcept
    {
        return parent_;
    }

    /// Access the local name of this module within its parent.
    const std::string& name() const noexcept
    {
        return name_;
    }

protected:
    Module*     parent_ = nullptr;
    std::string name_;

    /// Helper for add_child() overrides: sets parent and name on a child module.
    /// This is needed because derived classes cannot access protected members
    /// of other Module instances through the C++ protected access rules.
    static void attach_child_(Module* child, Module* parent, std::string name)
    {
        child->parent_ = parent;
        child->name_   = std::move(name);
    }
};

// ======================================================================
// ModuleList — indexed container for layer/expert sequences
// ======================================================================

/// A systematic container for indexed module sequences (layers, experts).
/// Children are added explicitly via ``add_child`` or ``create_child``.
class ModuleList: public Module {
public:
    const char* type() const override
    {
        return "ModuleList";
    }

    ModuleList() = default;

    explicit ModuleList(const core::ModuleListConfig&) {}  // empty config, no-op

    /// Override to also track the child in the indexed_ vector.
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;

    /// Find child by name.
    Module* child(const std::string& name) const override;

    /// Iterate over children.
    void for_each_child(std::function<void(const char*, Module*)> visitor) const override;

    /// Number of children created so far.
    int size() const;

private:
    std::vector<std::pair<std::string, std::unique_ptr<Module>>> items_;
    std::vector<Module*>                                         indexed_;
};

}  // namespace turbomind::core

#endif  // TURBOMIND_CORE_MODULE_H
