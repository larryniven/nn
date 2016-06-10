#ifndef LSTM_H
#define LSTM_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include <random>

namespace lstm {

    struct lstm_unit_param_t {
        la::matrix<double> hidden_input;
        la::matrix<double> hidden_output;
        la::vector<double> hidden_bias;

        la::matrix<double> input_input;
        la::matrix<double> input_output;
        la::vector<double> input_peep;
        la::vector<double> input_bias;

        la::matrix<double> output_input;
        la::matrix<double> output_output;
        la::vector<double> output_peep;
        la::vector<double> output_bias;

        la::matrix<double> forget_input;
        la::matrix<double> forget_output;
        la::vector<double> forget_peep;
        la::vector<double> forget_bias;
    };

    void imul(lstm_unit_param_t& param, double a);
    void iadd(lstm_unit_param_t& p1, lstm_unit_param_t const& p2);

    lstm_unit_param_t load_lstm_unit_param(std::istream& is);
    lstm_unit_param_t load_lstm_unit_param(std::string filename);

    void save_lstm_unit_param(lstm_unit_param_t const& p, std::ostream& os);
    void save_lstm_unit_param(lstm_unit_param_t const& p, std::string filename);

    void const_step_update_momentum(lstm_unit_param_t& p, lstm_unit_param_t const& grad,
        lstm_unit_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(lstm_unit_param_t& p, lstm_unit_param_t const& grad,
        lstm_unit_param_t& opt_data, double step_size);

    void rmsprop_update(lstm_unit_param_t& p, lstm_unit_param_t const& grad,
        lstm_unit_param_t& opt_data, double decay, double step_size);

    struct lstm_unit_nn_t {
        std::shared_ptr<autodiff::op_t> hidden_input;
        std::shared_ptr<autodiff::op_t> hidden_output;
        std::shared_ptr<autodiff::op_t> hidden_bias;

        std::shared_ptr<autodiff::op_t> input_input;
        std::shared_ptr<autodiff::op_t> input_output;
        std::shared_ptr<autodiff::op_t> input_peep;
        std::shared_ptr<autodiff::op_t> input_bias;

        std::shared_ptr<autodiff::op_t> output_input;
        std::shared_ptr<autodiff::op_t> output_output;
        std::shared_ptr<autodiff::op_t> output_peep;
        std::shared_ptr<autodiff::op_t> output_bias;

        std::shared_ptr<autodiff::op_t> forget_input;
        std::shared_ptr<autodiff::op_t> forget_output;
        std::shared_ptr<autodiff::op_t> forget_peep;
        std::shared_ptr<autodiff::op_t> forget_bias;
    };

    lstm_unit_nn_t make_lstm_unit_nn(autodiff::computation_graph& g,
        lstm_unit_param_t const& p);

    lstm_unit_param_t copy_lstm_unit_grad(lstm_unit_nn_t const& nn);

    struct lstm_step_nn_t {
        std::shared_ptr<autodiff::op_t> input_gate;
        std::shared_ptr<autodiff::op_t> output_gate;
        std::shared_ptr<autodiff::op_t> forget_gate;
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> cell;
    };

    lstm_step_nn_t make_lstm_step(lstm_unit_nn_t const& unit_nn,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input);

    // 2-lstm

    struct lstm2d_param_t {
        lstm_unit_param_t h_param;
        lstm_unit_param_t v_param;

        la::matrix<double> output_h_weight;
        la::matrix<double> output_v_weight;
        la::vector<double> output_bias;
    };

    lstm2d_param_t load_lstm2d_param(std::istream& is);
    lstm2d_param_t load_lstm2d_param(std::string filename);

    void save_lstm2d_param(lstm2d_param_t const& p, std::ostream& os);
    void save_lstm2d_param(lstm2d_param_t const& p, std::string filename);

    void const_step_update_momentum(lstm2d_param_t& p, lstm2d_param_t const& grad,
        lstm2d_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(lstm2d_param_t& p, lstm2d_param_t const& grad,
        lstm2d_param_t& opt_data, double step_size);

    void rmsprop_update(lstm2d_param_t& p, lstm2d_param_t const& grad,
        lstm2d_param_t& opt_data, double decay, double step_size);

    struct lstm2d_nn_t {
        lstm_unit_nn_t h_nn;
        lstm_unit_nn_t v_nn;

        std::shared_ptr<autodiff::op_t> output_h_weight;
        std::shared_ptr<autodiff::op_t> output_v_weight;
        std::shared_ptr<autodiff::op_t> output_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> h_cell;
        std::vector<std::shared_ptr<autodiff::op_t>> h_output;
        std::vector<std::shared_ptr<autodiff::op_t>> v_cell;
        std::vector<std::shared_ptr<autodiff::op_t>> v_output;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    lstm2d_nn_t make_lstm2d_nn(autodiff::computation_graph& graph,
        lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    lstm2d_param_t copy_lstm2d_grad(lstm2d_nn_t const& nn);

    // bidirectional 2-lstm

    struct bi_lstm2d_param_t {
        lstm2d_param_t forward_param;
        lstm2d_param_t backward_param;
    };

    bi_lstm2d_param_t load_bi_lstm2d_param(std::istream& is);
    bi_lstm2d_param_t load_bi_lstm2d_param(std::string filename);

    void save_bi_lstm2d_param(bi_lstm2d_param_t const& p, std::ostream& os);
    void save_bi_lstm2d_param(bi_lstm2d_param_t const& p, std::string filename);

    void const_step_update_momentum(bi_lstm2d_param_t& p, bi_lstm2d_param_t const& grad,
        bi_lstm2d_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(bi_lstm2d_param_t& p, bi_lstm2d_param_t const& grad,
        bi_lstm2d_param_t& opt_data, double step_size);

    void rmsprop_update(bi_lstm2d_param_t& p, bi_lstm2d_param_t const& grad,
        bi_lstm2d_param_t& opt_data, double decay, double step_size);

    struct bi_lstm2d_nn_t {
        lstm2d_nn_t forward_nn;
        lstm2d_nn_t backward_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    bi_lstm2d_nn_t make_bi_lstm2d_nn(autodiff::computation_graph& graph,
        bi_lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> inputs);

    bi_lstm2d_param_t copy_bi_lstm2d_grad(bi_lstm2d_nn_t const& nn);

    // deep bidirectional 2-lstm

    struct db_lstm2d_param_t {
        std::vector<bi_lstm2d_param_t> layer;
    };

    db_lstm2d_param_t load_db_lstm2d_param(std::istream& is);
    db_lstm2d_param_t load_db_lstm2d_param(std::string filename);

    void save_db_lstm2d_param(db_lstm2d_param_t const& p, std::ostream& os);
    void save_db_lstm2d_param(db_lstm2d_param_t const& p, std::string filename);

    void const_step_update_momentum(db_lstm2d_param_t& p, db_lstm2d_param_t const& grad,
        db_lstm2d_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(db_lstm2d_param_t& p, db_lstm2d_param_t const& grad,
        db_lstm2d_param_t& opt_data, double step_size);

    void rmsprop_update(db_lstm2d_param_t& p, db_lstm2d_param_t const& grad,
        db_lstm2d_param_t& opt_data, double decay, double step_size);

    struct db_lstm2d_nn_t {
        std::vector<bi_lstm2d_nn_t> layer;
    };

    lstm2d_nn_t stack_lstm2d(autodiff::computation_graph& graph,
        lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs,
        std::vector<std::shared_ptr<autodiff::op_t>> const& v_output,
        std::vector<std::shared_ptr<autodiff::op_t>> const& v_cell);

    bi_lstm2d_nn_t stack_bi_lstm2d(autodiff::computation_graph& graph,
        bi_lstm2d_param_t const& param, bi_lstm2d_nn_t const& prev);

    db_lstm2d_nn_t make_db_lstm2d_nn(autodiff::computation_graph& graph,
        db_lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    db_lstm2d_param_t copy_db_lstm2d_grad(db_lstm2d_nn_t const& nn);

    // lstm

    struct lstm_feat_nn_t {
        std::shared_ptr<autodiff::op_t> hidden_input;
        std::shared_ptr<autodiff::op_t> hidden_output;
        std::shared_ptr<autodiff::op_t> hidden_bias;

        std::shared_ptr<autodiff::op_t> input_input;
        std::shared_ptr<autodiff::op_t> input_output;
        std::shared_ptr<autodiff::op_t> input_peep;
        std::shared_ptr<autodiff::op_t> input_bias;

        std::shared_ptr<autodiff::op_t> output_input;
        std::shared_ptr<autodiff::op_t> output_output;
        std::shared_ptr<autodiff::op_t> output_peep;
        std::shared_ptr<autodiff::op_t> output_bias;

        std::shared_ptr<autodiff::op_t> forget_input;
        std::shared_ptr<autodiff::op_t> forget_output;
        std::shared_ptr<autodiff::op_t> forget_peep;
        std::shared_ptr<autodiff::op_t> forget_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> cell;
        std::vector<std::shared_ptr<autodiff::op_t>> hidden;
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate;
        std::vector<std::shared_ptr<autodiff::op_t>> output_gate;
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate;
        std::vector<std::shared_ptr<autodiff::op_t>> output;

        std::shared_ptr<autodiff::op_t> input_mask;
        std::shared_ptr<autodiff::op_t> hidden_mask;
    };

    lstm_feat_nn_t make_forward_lstm_feat_nn(autodiff::computation_graph& g,
        lstm_unit_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    lstm_feat_nn_t make_backward_lstm_feat_nn(autodiff::computation_graph& g,
        lstm_unit_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    lstm_unit_param_t copy_lstm_feat_grad(lstm_feat_nn_t const& nn);

    struct blstm_feat_param_t {
        lstm_unit_param_t forward_param;
        lstm_unit_param_t backward_param;

        la::matrix<double> forward_output_weight;
        la::matrix<double> backward_output_weight;
        la::vector<double> output_bias;
    };

    void imul(blstm_feat_param_t& param, double a);
    void iadd(blstm_feat_param_t& p1, blstm_feat_param_t& p2);

    blstm_feat_param_t load_blstm_feat_param(std::istream& is);
    blstm_feat_param_t load_blstm_feat_param(std::string filename);

    void save_blstm_feat_param(blstm_feat_param_t const& p, std::ostream& os);
    void save_blstm_feat_param(blstm_feat_param_t const& p, std::string filename);

    void const_step_update_momentum(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
        blstm_feat_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
        blstm_feat_param_t& opt_data, double step_size);

    void rmsprop_update(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
        blstm_feat_param_t& opt_data, double decay, double step_size);

    struct blstm_feat_nn_t {
        lstm_feat_nn_t forward_feat_nn;
        lstm_feat_nn_t backward_feat_nn;

        std::shared_ptr<autodiff::op_t> forward_output_weight;
        std::shared_ptr<autodiff::op_t> backward_output_weight;
        std::shared_ptr<autodiff::op_t> output_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    blstm_feat_nn_t make_blstm_feat_nn(autodiff::computation_graph& g,
        blstm_feat_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    blstm_feat_param_t copy_blstm_feat_grad(blstm_feat_nn_t const& nn);

    struct dblstm_feat_param_t {
        std::vector<blstm_feat_param_t> layer;
    };

    void imul(dblstm_feat_param_t& param, double a);
    void iadd(dblstm_feat_param_t& p1, dblstm_feat_param_t& p2);

    dblstm_feat_param_t load_dblstm_feat_param(std::istream& is);
    dblstm_feat_param_t load_dblstm_feat_param(std::string filename);

    void save_dblstm_feat_param(dblstm_feat_param_t const& p, std::ostream& os);
    void save_dblstm_feat_param(dblstm_feat_param_t const& p, std::string filename);

    void const_step_update_momentum(dblstm_feat_param_t& p, dblstm_feat_param_t const& grad,
        dblstm_feat_param_t& opt_data, double momentum, double step_size);

    void adagrad_update(dblstm_feat_param_t& p, dblstm_feat_param_t const& grad,
        dblstm_feat_param_t& opt_data, double step_size);

    void rmsprop_update(dblstm_feat_param_t& p, dblstm_feat_param_t const& grad,
        dblstm_feat_param_t& opt_data, double decay, double step_size);

    struct dblstm_feat_nn_t {
        std::vector<blstm_feat_nn_t> layer;
    };

    dblstm_feat_nn_t make_dblstm_feat_nn(autodiff::computation_graph& graph,
        dblstm_feat_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

    dblstm_feat_param_t copy_dblstm_feat_grad(dblstm_feat_nn_t const& nn);

    void apply_random_mask(dblstm_feat_nn_t& nn, dblstm_feat_param_t const& param,
        std::default_random_engine& gen, double prob);

    void apply_mask(dblstm_feat_nn_t& nn, dblstm_feat_param_t const& param, double prob);

}

#endif
