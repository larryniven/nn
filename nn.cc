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
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

        root.children.push_back(tensor_tree::make_tensor("softmax weight"));
        root.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    log_loss::log_loss(la::tensor_like<double> const& gold, la::tensor_like<double> const& pred)
        : gold(gold), pred(pred)
    {}
    
    double log_loss::loss()
    {
        return -la::dot(gold, pred);
    }
    
    la::tensor<double> log_loss::grad(double scale)
    {
        return la::mul(gold, -scale);
    }

    l2_loss::l2_loss(la::tensor_like<double> const& gold, la::tensor_like<double> const& pred)
        : gold(gold), pred(pred)
    {}
    
    double l2_loss::loss()
    {
        la::tensor<double> diff;
        diff.resize(gold.sizes());
        la::copy(diff, gold);
        la::isub(diff, pred);

        return la::dot(diff, diff);
    }
    
    la::tensor<double> l2_loss::grad(double scale)
    {
        la::tensor<double> g = la::mul(pred, 2 * scale);
        la::axpy(g, -2 * scale, gold);

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

