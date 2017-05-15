#include "nn/nn.h"
#include <fstream>

namespace nn {

    pred_nn_t make_pred_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        pred_nn_t result;

        result.logprob = autodiff::logsoftmax(
            autodiff::add(autodiff::mul(input, tensor_tree::get_var(var_tree->children[0])),
                tensor_tree::get_var(var_tree->children[1])));

        return result;
    }

    std::shared_ptr<tensor_tree::vertex> make_pred_tensor_tree()
    {
        tensor_tree::vertex root { "nil" };

        root.children.push_back(tensor_tree::make_tensor("softmax weight"));
        root.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    log_loss::log_loss(la::cpu::tensor_like<double> const& gold, la::cpu::tensor_like<double> const& pred)
        : gold(gold), pred(pred)
    {}
    
    double log_loss::loss()
    {
        return -la::cpu::dot(gold, pred);
    }
    
    la::cpu::tensor<double> log_loss::grad(double scale)
    {
        return la::cpu::mul(gold, -scale);
    }

    l2_loss::l2_loss(la::cpu::tensor_like<double> const& gold, la::cpu::tensor_like<double> const& pred)
        : gold(gold), pred(pred)
    {}
    
    double l2_loss::loss()
    {
        la::cpu::tensor<double> diff;
        diff.resize(gold.sizes());
        la::cpu::copy(diff, gold);
        la::cpu::isub(diff, pred);

        return la::cpu::dot(diff, diff);
    }
    
    la::cpu::tensor<double> l2_loss::grad(double scale)
    {
        la::cpu::tensor<double> g = la::cpu::mul(pred, 2 * scale);
        la::cpu::axpy(g, -2 * scale, gold);

        return g;
    }

    seq_pred_nn_t make_seq_pred_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat)
    {
        seq_pred_nn_t result;

        for (int i = 0; i < feat.size(); ++i) {
            result.logprob.push_back(autodiff::logsoftmax(autodiff::add(
                autodiff::mul(feat[i], tensor_tree::get_var(var_tree->children[0])),
                    tensor_tree::get_var(var_tree->children[1]))));
        }

        return result;
    }

}

