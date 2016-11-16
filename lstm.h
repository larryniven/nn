#ifndef LSTM_H
#define LSTM_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include <random>
#include "nn/tensor_tree.h"

namespace lstm {

    // lstm

    struct lstm_tensor_tree_factory {

        virtual ~lstm_tensor_tree_factory();

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    struct dyer_lstm_tensor_tree_factory
        : public lstm_tensor_tree_factory {

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree();

    struct lstm_step_nn_t {
        std::shared_ptr<autodiff::op_t> input_gate;
        std::shared_ptr<autodiff::op_t> output_gate;
        std::shared_ptr<autodiff::op_t> forget_gate;
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> cell;
        std::shared_ptr<autodiff::op_t> input;
    };

    lstm_step_nn_t make_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input);

    lstm_step_nn_t make_dyer_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input);

    struct lstm_nn_t {
        std::vector<std::shared_ptr<autodiff::op_t>> cell;
        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    struct lstm_builder {

        virtual ~lstm_builder();

        virtual lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;

    };

    struct dyer_lstm_builder
        : public lstm_builder {

        std::shared_ptr<autodiff::op_t> one;

        dyer_lstm_builder(std::shared_ptr<autodiff::op_t> one);

        virtual lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;
    };

    // bidirectional lstm

    struct bi_lstm_tensor_tree_factory {

        std::shared_ptr<lstm_tensor_tree_factory> base_fac;

        bi_lstm_tensor_tree_factory();
        bi_lstm_tensor_tree_factory(std::shared_ptr<lstm_tensor_tree_factory> base_fac);

        virtual ~bi_lstm_tensor_tree_factory();

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    std::shared_ptr<tensor_tree::vertex> make_bi_lstm_tensor_tree();

    struct bi_lstm_nn_t {
        lstm_nn_t forward_nn;
        lstm_nn_t backward_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    bi_lstm_nn_t make_bi_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        lstm_builder const& builder);

    struct bi_lstm_builder {

        virtual ~bi_lstm_builder();

        virtual bi_lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;
    };

    struct dyer_bi_lstm_builder
        : public bi_lstm_builder {

        std::shared_ptr<autodiff::op_t> one;

        dyer_bi_lstm_builder(std::shared_ptr<autodiff::op_t> one);

        virtual bi_lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;
    };

    struct bi_lstm_input_dropout
        : public bi_lstm_builder {

        std::default_random_engine& gen;
        double prob;
        std::shared_ptr<bi_lstm_builder> builder;

        bi_lstm_input_dropout(std::default_random_engine& gen, double prob,
            std::shared_ptr<bi_lstm_builder> builder);

        virtual bi_lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;
    };

    struct bi_lstm_input_scaling
        : public bi_lstm_builder {

        double scale;
        std::shared_ptr<bi_lstm_builder> builder;

        bi_lstm_input_scaling(double scale,
            std::shared_ptr<bi_lstm_builder> builder);

        virtual bi_lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;
    };

    struct bi_lstm_input_subsampling
        : public bi_lstm_builder {

        std::shared_ptr<bi_lstm_builder> builder;
        mutable int freq;
        mutable bool once;

        bi_lstm_input_subsampling(std::shared_ptr<bi_lstm_builder> builder);

        virtual bi_lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;
    };

    // stacked bidirectional lstm

    struct stacked_bi_lstm_tensor_tree_factory {

        int layer;
        std::shared_ptr<bi_lstm_tensor_tree_factory> base_fac;

        stacked_bi_lstm_tensor_tree_factory(int layer);
        stacked_bi_lstm_tensor_tree_factory(int layer, std::shared_ptr<bi_lstm_tensor_tree_factory> base_fac);

        virtual ~stacked_bi_lstm_tensor_tree_factory();

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    std::shared_ptr<tensor_tree::vertex> make_stacked_bi_lstm_tensor_tree(int layer);

    struct stacked_bi_lstm_nn_t {
        std::vector<bi_lstm_nn_t> layer;
    };

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        bi_lstm_builder const& builder);

    std::vector<std::string> subsample(
        std::vector<std::string> const& input,
        int freq, int shift);

    std::vector<std::shared_ptr<autodiff::op_t>> subsample(
        std::vector<std::shared_ptr<autodiff::op_t>> const& input,
        int freq, int shift);

    std::vector<std::shared_ptr<autodiff::op_t>> upsample(
        std::vector<std::shared_ptr<autodiff::op_t>> const& input,
        int freq, int shift, int length);

    lstm_nn_t make_zoneout_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask,
        std::shared_ptr<autodiff::op_t> one);

    struct zoneout_lstm_builder
        : public lstm_builder {

        std::vector<std::shared_ptr<autodiff::op_t>> mask;
        std::shared_ptr<autodiff::op_t> one;

        zoneout_lstm_builder(std::vector<std::shared_ptr<autodiff::op_t>> const& mask,
            std::shared_ptr<autodiff::op_t> one);

        virtual lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;
    };

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
