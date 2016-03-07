#ifndef GRU_H
#define GRU_H

#include "la/la.h"
#include "autodiff/autodiff.h"

namespace gru {

    struct gru_feat_param_t {
        la::matrix<double> reset_input;
        la::matrix<double> reset_hidden;
        la::vector<double> reset_bias;

        la::matrix<double> update_input;
        la::matrix<double> update_hidden;
        la::vector<double> update_bias;

        la::matrix<double> candidate_input;
        la::matrix<double> candidate_hidden;
        la::vector<double> candidate_bias;

        la::matrix<double> shortcut_input;
        la::vector<double> shortcut_bias;
    };

    gru_feat_param_t load_gru_feat_param(std::istream& is);
    gru_feat_param_t load_gru_feat_param(std::string filename);

    void save_gru_feat_param(gru_feat_param_t const& param, std::ostream& os);
    void save_gru_feat_param(gru_feat_param_t const& param, std::string filename);

    void adagrad_update(gru_feat_param_t& param, gru_feat_param_t const& grad,
        gru_feat_param_t& opt_data, double step_size);

    void rmsprop_update(gru_feat_param_t& param, gru_feat_param_t const& grad,
        gru_feat_param_t& opt_data, double decay, double step_size);

    struct gru_feat_nn_t {
        std::shared_ptr<autodiff::op_t> reset_input;
        std::shared_ptr<autodiff::op_t> reset_hidden;
        std::shared_ptr<autodiff::op_t> reset_bias;

        std::shared_ptr<autodiff::op_t> update_input;
        std::shared_ptr<autodiff::op_t> update_hidden;
        std::shared_ptr<autodiff::op_t> update_bias;

        std::shared_ptr<autodiff::op_t> candidate_input;
        std::shared_ptr<autodiff::op_t> candidate_hidden;
        std::shared_ptr<autodiff::op_t> candidate_bias;

        std::shared_ptr<autodiff::op_t> shortcut_input;
        std::shared_ptr<autodiff::op_t> shortcut_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> reset;
        std::vector<std::shared_ptr<autodiff::op_t>> update;
        std::vector<std::shared_ptr<autodiff::op_t>> hidden;
        std::vector<std::shared_ptr<autodiff::op_t>> candidate;

        std::shared_ptr<autodiff::op_t> one;
    };

    gru_feat_nn_t make_gru_feat_nn(autodiff::computation_graph& g,
        gru_feat_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    gru_feat_param_t copy_grad(gru_feat_nn_t const& nn);

    struct bgru_feat_param_t {
        gru_feat_param_t forward_param;
        gru_feat_param_t backward_param;

        la::matrix<double> forward_output;
        la::matrix<double> backward_output;
        la::vector<double> output_bias;
    };

    bgru_feat_param_t load_bgru_feat_param(std::istream& is);
    bgru_feat_param_t load_bgru_feat_param(std::string filename);

    void save_bgru_feat_param(bgru_feat_param_t const& param, std::ostream& os);
    void save_bgru_feat_param(bgru_feat_param_t const& param, std::string filename);

    void adagrad_update(bgru_feat_param_t& param, bgru_feat_param_t const& grad,
        bgru_feat_param_t& opt_data, double step_size);

    void rmsprop_update(bgru_feat_param_t& param, bgru_feat_param_t const& grad,
        bgru_feat_param_t& opt_data, double decay, double step_size);

    struct bgru_feat_nn_t {
        std::shared_ptr<autodiff::op_t> forward_output;
        std::shared_ptr<autodiff::op_t> backward_output;
        std::shared_ptr<autodiff::op_t> output_bias;

        gru_feat_nn_t forward_nn;
        gru_feat_nn_t backward_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    bgru_feat_nn_t make_bgru_feat_nn(autodiff::computation_graph& g,
        bgru_feat_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    bgru_feat_param_t copy_grad(bgru_feat_nn_t const& nn);

    struct dbgru_feat_param_t {
        std::vector<bgru_feat_param_t> layer;
    };

    dbgru_feat_param_t load_dbgru_feat_param(std::istream& is);
    dbgru_feat_param_t load_dbgru_feat_param(std::string filename);

    void save_dbgru_feat_param(dbgru_feat_param_t const& param, std::ostream& os);
    void save_dbgru_feat_param(dbgru_feat_param_t const& param, std::string filename);

    void adagrad_update(dbgru_feat_param_t& param, dbgru_feat_param_t const& grad,
        dbgru_feat_param_t& opt_data, double step_size);

    void rmsprop_update(dbgru_feat_param_t& param, dbgru_feat_param_t const& grad,
        dbgru_feat_param_t& opt_data, double decay, double step_size);

    struct dbgru_feat_nn_t {
        std::vector<bgru_feat_nn_t> layer;
    };

    dbgru_feat_nn_t make_dbgru_feat_nn(autodiff::computation_graph& g,
        dbgru_feat_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    dbgru_feat_param_t copy_grad(dbgru_feat_nn_t const& nn);

    struct pred_param_t {
        la::matrix<double> softmax_weight;
        la::vector<double> softmax_bias;
    };

    pred_param_t load_pred_param(std::istream& is);
    pred_param_t load_pred_param(std::string filename);

    void save_pred_param(pred_param_t const& param, std::ostream& os);
    void save_pred_param(pred_param_t const& param, std::string filename);

    void adagrad_update(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double step_size);

    void rmsprop_update(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double decay, double step_size);

    struct pred_nn_t {
        std::shared_ptr<autodiff::op_t> softmax_weight;
        std::shared_ptr<autodiff::op_t> softmax_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> logprob;
    };

    pred_nn_t make_pred_nn(autodiff::computation_graph& g,
        pred_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

    pred_param_t copy_grad(pred_nn_t const& nn);

    void eval(pred_nn_t& nn);
    void grad(pred_nn_t& nn);

    struct log_loss {

        la::vector<double> gold;
        la::vector<double> pred;

        double loss();

        la::vector<double> grad();

    };

}

#endif
