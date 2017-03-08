#ifndef NN_H
#define NN_H

#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"

namespace nn {

    struct pred_nn_t {
        std::shared_ptr<autodiff::op_t> logprob;
    };

    pred_nn_t make_pred_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input);

    std::shared_ptr<tensor_tree::vertex> make_pred_tensor_tree();

    struct log_loss {
    
        la::tensor_like<double> const& gold;
        la::tensor_like<double> const& pred;
    
        log_loss(la::tensor_like<double> const& gold,
            la::tensor_like<double> const& pred);
    
        double loss();
    
        la::tensor<double> grad(double scale=1);
    
    };

    struct l2_loss {

        la::tensor_like<double> const& gold;
        la::tensor_like<double> const& pred;
    
        l2_loss(la::tensor_like<double> const& gold,
            la::tensor_like<double> const& pred);
    
        double loss();
    
        la::tensor<double> grad(double scale=1);
    
    };

    struct seq_pred_nn_t {
        std::vector<std::shared_ptr<autodiff::op_t>> logprob;
    };

    seq_pred_nn_t make_seq_pred_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

}

#endif
