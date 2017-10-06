#include "nn/autoenc-fc.h"

namespace nn {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        tensor_tree::vertex root { "nil" };
    
        for (int i = 0; i < layer; ++i) {
            tensor_tree::vertex fc;
            fc.children.push_back(tensor_tree::make_tensor("fc weight"));
            fc.children.push_back(tensor_tree::make_tensor("fc bias"));
            root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));
        }
    
        tensor_tree::vertex pred;
        pred.children.push_back(tensor_tree::make_tensor("fc weight"));
        pred.children.push_back(tensor_tree::make_tensor("fc bias"));
        root.children.push_back(std::make_shared<tensor_tree::vertex>(pred));
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<autodiff::op_t> make_nn(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        double input_dropout, double hidden_dropout,
        std::default_random_engine& gen)
    {
        std::shared_ptr<autodiff::op_t> h = input;

        if (input_dropout != 0.0) {
            auto mask = autodiff::dropout_mask(h, input_dropout, gen);
            h = autodiff::emul(mask, h);
        }
    
        for (int i = 0; i < var_tree->children.size() - 1; ++i) {
            auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children[i]->children[0]));
            auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[i]->children[1]), z);
            h = autodiff::relu(autodiff::add(z, b));

            if (hidden_dropout != 0.0) {
                auto mask = autodiff::dropout_mask(h, hidden_dropout, gen);
                h = autodiff::emul(mask, h);
            }
        }
    
        auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children.back()->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children.back()->children[1]), z);
    
        return autodiff::add(z, b);
    }

    std::shared_ptr<autodiff::op_t> make_nn(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        int k)
    {
        std::shared_ptr<autodiff::op_t> h = input;
    
        for (int i = 0; i < var_tree->children.size() - 1; ++i) {
            auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children[i]->children[0]));
            auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[i]->children[1]), z);
            h = autodiff::relu(autodiff::add(z, b));
            h = autodiff::high_pass_k(h, k);
        }
    
        auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children.back()->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children.back()->children[1]), z);
    
        return autodiff::add(z, b);
    }

}
