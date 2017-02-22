#include "nn/rsg.h"
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"

namespace rsg {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        lstm::dyer_lstm_tensor_tree_factory fac;

        tensor_tree::vertex root;

        root.children.push_back(tensor_tree::make_tensor("label weight"));
        root.children.push_back(tensor_tree::make_tensor("duration weight"));
        root.children.push_back(tensor_tree::make_tensor("acoustic weight"));
        root.children.push_back(tensor_tree::make_tensor("input bias"));

        tensor_tree::vertex lstm_step;

        for (int i = 0; i < layer; ++i) {
            lstm_step.children.push_back(fac());
        }

        root.children.push_back(std::make_shared<tensor_tree::vertex>(lstm_step));
        root.children.push_back(tensor_tree::make_tensor("output weight"));
        root.children.push_back(tensor_tree::make_tensor("output bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::vector<std::shared_ptr<autodiff::op_t>> make_training_nn(
        std::shared_ptr<autodiff::op_t> init_cell,
        std::vector<std::shared_ptr<autodiff::op_t>> const& gt_seq,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<lstm::step_transcriber> step)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        std::shared_ptr<autodiff::op_t> cell = autodiff::add(
            autodiff::mul(init_cell, tensor_tree::get_var(var_tree->children[0])),
            tensor_tree::get_var(var_tree->children[1]));

        std::shared_ptr<autodiff::op_t> output = nullptr;

        for (int t = 0; t < gt_seq.size(); ++t) {
            auto output = (*step)(var_tree->children[2], gt_seq.at(t));

            result.push_back(autodiff::add(
                autodiff::mul(output, tensor_tree::get_var(var_tree->children[3])),
                tensor_tree::get_var(var_tree->children[4])));
        }

        return result;
    }

    std::vector<std::shared_ptr<autodiff::op_t>> make_nn(
        std::shared_ptr<autodiff::op_t> init_cell,
        std::shared_ptr<autodiff::op_t> init_input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        int steps,
        std::shared_ptr<lstm::step_transcriber> step)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        std::shared_ptr<autodiff::op_t> cell = autodiff::add(
            autodiff::mul(init_cell, tensor_tree::get_var(var_tree->children[0])),
            tensor_tree::get_var(var_tree->children[1]));

        std::shared_ptr<autodiff::op_t> output = nullptr;
        std::shared_ptr<autodiff::op_t> input = init_input;

        for (int t = 0; t < steps; ++t) {
            auto output = (*step)(var_tree->children[2], input);

            std::shared_ptr<autodiff::op_t> target = autodiff::add(
                autodiff::mul(output, tensor_tree::get_var(var_tree->children[3])),
                tensor_tree::get_var(var_tree->children[4]));

            input = target;

            result.push_back(target);
        }

        return result;
    }
}
