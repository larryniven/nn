#ifndef LSTM_SEG_H
#define LSTM_SEG_H

#include <memory>
#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"
#include "nn/lstm.h"

namespace lstm_seg {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer,
        std::unordered_map<std::string, std::string> const& args);

    std::shared_ptr<autodiff::op_t> make_pred_nn(
        autodiff::computation_graph& graph,
        lstm::stacked_bi_lstm_nn_t& nn,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<tensor_tree::vertex> param,
        std::unordered_map<std::string, std::string> const& args);

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

    std::shared_ptr<autodiff::op_t> make_pred_nn(
        autodiff::computation_graph& graph,
        lstm::stacked_bi_lstm_nn_t& nn,
        std::shared_ptr<tensor_tree::vertex> var_tree);

    std::shared_ptr<autodiff::op_t> make_pred_nn_uniform(
        autodiff::computation_graph& graph,
        lstm::stacked_bi_lstm_nn_t& nn,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        int h_dim);

    namespace endpoints {

        std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

        std::shared_ptr<autodiff::op_t> make_pred_nn(
            autodiff::computation_graph& graph,
            lstm::stacked_bi_lstm_nn_t& nn,
            std::shared_ptr<tensor_tree::vertex> var_tree);

    }

    namespace logp {

        std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

        std::shared_ptr<autodiff::op_t> make_pred_nn(
            autodiff::computation_graph& graph,
            lstm::stacked_bi_lstm_nn_t& nn,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            int h_dim);

    }

}

#endif
