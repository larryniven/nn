#include "nn/rhn.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace rhn {

    std::shared_ptr<tensor_tree::vertex> make_rhn_tensor_tree()
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

        // 0
        root.children.push_back(tensor_tree::make_matrix("input -> hidden"));
        root.children.push_back(tensor_tree::make_matrix("cell -> hidden"));
        root.children.push_back(tensor_tree::make_vector("hidden bias"));

        // 3
        root.children.push_back(tensor_tree::make_matrix("input -> input gate"));
        root.children.push_back(tensor_tree::make_matrix("cell -> input gate"));
        root.children.push_back(tensor_tree::make_vector("input gate bias"));

        // 6
        // root.children.push_back(tensor_tree::make_matrix("input -> forget gate"));
        // root.children.push_back(tensor_tree::make_matrix("cell -> forget gate"));
        // root.children.push_back(tensor_tree::make_vector("forget gate bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<autodiff::op_t> make_rhn_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> input)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> h_comp { get_var(var_tree->children[2]) };
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp { get_var(var_tree->children[5]) };
        // std::vector<std::shared_ptr<autodiff::op_t>> forget_gate_comp { get_var(var_tree->children[8]) };

        if (input != nullptr) {
            h_comp.push_back(autodiff::mul(get_var(var_tree->children[0]), input));
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[3]), input));
            // forget_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[6]), input));
        }

        if (cell != nullptr) {
            h_comp.push_back(autodiff::mul(get_var(var_tree->children[1]), cell));
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[4]), cell));
            // forget_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[7]), cell));
        }

        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));
        std::shared_ptr<autodiff::op_t> input_gate = autodiff::logistic(autodiff::add(input_gate_comp));
        // std::shared_ptr<autodiff::op_t> forget_gate = autodiff::logistic(autodiff::add(forget_gate_comp));

        if (cell != nullptr) {
            return autodiff::add(
                autodiff::sub(cell, autodiff::emul(input_gate, cell)),
                autodiff::emul(input_gate, h));
        } else {
            return autodiff::emul(input_gate, h);
        }
    }

    rhn_nn_t make_rhn_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        int depth)
    {
        rhn_nn_t result;

        std::shared_ptr<autodiff::op_t> cell = nullptr;
        result.cell.push_back(cell);

        for (int i = 0; i < feat.size(); ++i) {
            cell = make_rhn_step_nn(var_tree, cell, feat[i]);
            result.cell.push_back(cell);

            for (int j = 0; j < depth - 1; ++j) {
                cell = make_rhn_step_nn(var_tree, cell, nullptr);
                result.cell.push_back(cell);
            }

            result.output.push_back(cell);
        }

        return result;
    }

    rhn_nn_t rhn_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        int depth) const
    {
        return make_rhn_nn(var_tree, feat, depth);
    }

}
