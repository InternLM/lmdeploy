
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

class Module {
public:
    virtual ~Module();

    Module();

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    Module(Module&&) noexcept = delete;
    Module& operator=(Module&&) noexcept = delete;

    void register_module(std::string name, Module& module, std::optional<int> index = {});
    void register_parameter(std::string name, Tensor& param);

    void remove_module(Module& module);
    void remove_parameter(Tensor& param);

    std::unordered_map<std::string, Tensor*> get_parameters() const;

private:
    void get_parameters_impl(std::string prefix, std::unordered_map<std::string, Tensor*>& m) const;

protected:
    Module* parent_;

    std::vector<std::pair<std::string, Module*>> modules_;
    std::vector<std::pair<std::string, Tensor*>> params_;
};

}  // namespace turbomind::core
