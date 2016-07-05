#ifndef LSTM_H
#define LSTM_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include <random>
#include "nn/tensor_tree.h"

namespace lstm {

    // lstm

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree();

    struct lstm_step_nn_t {
        std::shared_ptr<autodiff::op_t> input_gate;
        std::shared_ptr<autodiff::op_t> output_gate;
        std::shared_ptr<autodiff::op_t> forget_gate;
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> cell;
    };

    lstm_step_nn_t make_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input);

    struct lstm_nn_t {
        std::vector<std::shared_ptr<autodiff::op_t>> cell;
        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    lstm_nn_t make_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

    std::shared_ptr<tensor_tree::vertex> make_bi_lstm_tensor_tree();

    // bidirectional lstm

    struct bi_lstm_nn_t {
        lstm_nn_t forward_nn;
        lstm_nn_t backward_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    bi_lstm_nn_t make_bi_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

    // stacked bidirectional lstm

    std::shared_ptr<tensor_tree::vertex> make_stacked_bi_lstm_tensor_tree(int layer);

    struct stacked_bi_lstm_nn_t {
        std::vector<bi_lstm_nn_t> layer;
    };

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat);

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn_with_dropout(
        autodiff::computation_graph& g,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::default_random_engine& gen, double prob);

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn_with_dropout(
        autodiff::computation_graph& g,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        double prob);

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn_with_dropout_light(
        autodiff::computation_graph& g,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::default_random_engine& gen, double prob);

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn_with_dropout_light(
        autodiff::computation_graph& g,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        double prob);

    std::vector<std::string> subsample(
        std::vector<std::string> const& input,
        int freq, int shift);

    std::vector<std::shared_ptr<autodiff::op_t>> subsample(
        std::vector<std::shared_ptr<autodiff::op_t>> const& input,
        int freq, int shift);

    std::vector<std::shared_ptr<autodiff::op_t>> upsample(
        std::vector<std::shared_ptr<autodiff::op_t>> const& input,
        int freq, int shift, int length);

#if 0
    struct lstm2d_nn_t {
        std::vector<std::shared_ptr<autodiff::op_t>> h_cell;
        std::vector<std::shared_ptr<autodiff::op_t>> h_output;
        std::vector<std::shared_ptr<autodiff::op_t>> v_cell;
        std::vector<std::shared_ptr<autodiff::op_t>> v_output;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    struct bi_lstm2d_nn_t {
        lstm2d_nn_t forward_nn;
        lstm2d_nn_t backward_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    struct db_lstm2d_nn_t {
        std::vector<bi_lstm2d_nn_t> layer;
    };

#endif

}

#endif
