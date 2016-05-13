#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include "opt/opt.h"

namespace residual {

    // x -------------------------------------------------- add -- 
    //     \                                          /
    //      ---- relu -- lin -- add -- relu -- lin ---
    //                           |
    //                          lin
    //                           |
    //                           y

    struct unit_param_t {
        la::matrix<double> weight1;
        la::vector<double> bias1;
        la::matrix<double> weight2;
        la::vector<double> bias2;
        // la::matrix<double> input_weight;
        // la::vector<double> input_bias;
    };

    unit_param_t load_unit_param(std::istream& is);
    void save_unit_param(unit_param_t const& p, std::ostream& os);

    void adagrad_update(unit_param_t& param, unit_param_t const& grad,
        unit_param_t& accu_grad_sq, double step_size);

    struct nn_unit_t {
        std::shared_ptr<autodiff::op_t> weight1;
        std::shared_ptr<autodiff::op_t> bias1;
        std::shared_ptr<autodiff::op_t> weight2;
        std::shared_ptr<autodiff::op_t> bias2;
        // std::shared_ptr<autodiff::op_t> input_weight;
        // std::shared_ptr<autodiff::op_t> input_bias;

        std::shared_ptr<autodiff::op_t> cell;
        std::shared_ptr<autodiff::op_t> output;
    };

    nn_unit_t make_unit_nn(autodiff::computation_graph& graph,
        unit_param_t const& param);

    unit_param_t copy_unit_grad(nn_unit_t const& unit);

    struct nn_param_t {
        std::vector<unit_param_t> layer;

        la::matrix<double> input_weight;
        la::vector<double> input_bias;
        la::matrix<double> softmax_weight;
        la::vector<double> softmax_bias;
    };

    nn_param_t load_nn_param(std::istream& is);
    void save_nn_param(nn_param_t const& p, std::ostream& os);

    void adagrad_update(nn_param_t& param, nn_param_t const& grad,
        nn_param_t& accu_grad_sq, double step_size);

    struct nn_t {
        std::vector<nn_unit_t> layer;

        std::shared_ptr<autodiff::op_t> input_weight;
        std::shared_ptr<autodiff::op_t> input_bias;
    };

    nn_t make_nn(autodiff::computation_graph& graph,
        nn_param_t const& param);

    nn_param_t copy_nn_grad(nn_t const& nn);

}

#endif
