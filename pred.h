#ifndef PRED_H
#define PRED_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include "nn/tensor_tree.h"

namespace nn {

    struct pred_param_t {
        la::matrix<double> softmax_weight;
        la::vector<double> softmax_bias;
    };

    pred_param_t load_pred_param(std::istream& is);
    pred_param_t load_pred_param(std::string filename);

    void save_pred_param(pred_param_t const& param, std::ostream& os);
    void save_pred_param(pred_param_t const& param, std::string filename);

    void const_step_update(pred_param_t& param, pred_param_t const& grad,
        double step_size);

    void const_step_update_momentum(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double step_size);

    void rmsprop_update(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double decay, double step_size);

    struct pred_nn_t {
        std::shared_ptr<autodiff::op_t> softmax_weight;
        std::shared_ptr<autodiff::op_t> softmax_bias;

        std::shared_ptr<autodiff::op_t> input;
        std::shared_ptr<autodiff::op_t> logprob;
    };

    pred_nn_t make_pred_nn(autodiff::computation_graph& g,
        std::shared_ptr<autodiff::op_t> input,
        pred_param_t const& param);

    pred_param_t copy_grad(pred_nn_t const& nn);

    std::shared_ptr<tensor_tree::vertex> make_pred_tensor_tree();

}

namespace rnn {

    struct pred_nn_t {
        std::shared_ptr<autodiff::op_t> softmax_weight;
        std::shared_ptr<autodiff::op_t> softmax_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> logprob;
    };

    pred_nn_t make_pred_nn(autodiff::computation_graph& g,
        nn::pred_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

    nn::pred_param_t copy_grad(pred_nn_t const& nn);

    void eval(pred_nn_t& nn);
    void grad(pred_nn_t& nn);

    std::vector<std::shared_ptr<autodiff::op_t>> subsample_input(
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs,
        int freq, int shift);
    
    std::vector<std::shared_ptr<autodiff::op_t>> upsample_output(
        std::vector<std::shared_ptr<autodiff::op_t>> const& outputs,
        int freq, int shift, int size);

    pred_nn_t make_pred_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

}

#endif
