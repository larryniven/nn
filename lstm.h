#ifndef LSTM_H
#define LSTM_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include <random>
#include "nn/tensor-tree.h"

namespace lstm {

    // lstm

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

        virtual lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;
    };

    struct multilayer_lstm_builder
        : public lstm_builder {

        std::shared_ptr<lstm_builder> builder;
        int layer;

        multilayer_lstm_builder(std::shared_ptr<lstm_builder> builder, int layer);

        virtual lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const;
    };

    // bidirectional lstm

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

        autodiff::computation_graph& comp_graph;
        int dim;
        std::default_random_engine& gen;
        double prob;
        std::shared_ptr<bi_lstm_builder> builder;

        bi_lstm_input_dropout(autodiff::computation_graph& comp_graph,
            int dim,
            std::default_random_engine& gen, double prob,
            std::shared_ptr<bi_lstm_builder> builder);

        virtual bi_lstm_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;
    };

    struct bi_lstm_input_scaling
        : public bi_lstm_builder {

        autodiff::computation_graph& comp_graph;
        int dim;
        double scale;
        std::shared_ptr<bi_lstm_builder> builder;

        bi_lstm_input_scaling(autodiff::computation_graph& comp_graph,
            int dim,
            double scale,
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

    struct lstm_step_transcriber {

        virtual ~lstm_step_transcriber();

        virtual lstm_step_nn_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> cell,
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> input) const;

    };

    struct lstm_input_dropout_transcriber
        : public lstm_step_transcriber {

        std::default_random_engine& gen;
        double prob;
        std::shared_ptr<lstm_step_transcriber> base;

        lstm_input_dropout_transcriber(std::default_random_engine& gen, double prob,
            std::shared_ptr<lstm_step_transcriber> base);

        virtual lstm_step_nn_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> cell,
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> input) const;
    };

    struct lstm_output_dropout_transcriber
        : public lstm_step_transcriber {

        std::default_random_engine& gen;
        double prob;
        std::shared_ptr<lstm_step_transcriber> base;

        lstm_output_dropout_transcriber(std::default_random_engine& gen, double prob,
            std::shared_ptr<lstm_step_transcriber> base);

        virtual lstm_step_nn_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> cell,
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> input) const;
    };

    struct dyer_lstm_step_transcriber
        : public lstm_step_transcriber {

        virtual lstm_step_nn_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> cell,
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> input) const override;

    };

    // tanscriber

    struct transcriber {
        virtual ~transcriber();

        virtual std::vector<std::shared_ptr<autodiff::op_t>> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const = 0;
    };

    struct lstm_transcriber
        : public transcriber {

        std::shared_ptr<lstm_step_transcriber> step;

        lstm_transcriber(std::shared_ptr<lstm_step_transcriber> step);

        virtual std::vector<std::shared_ptr<autodiff::op_t>> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;

    };

    struct bi_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;

        bi_transcriber(std::shared_ptr<transcriber> base);

        virtual std::vector<std::shared_ptr<autodiff::op_t>> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;
    };

    struct layered_transcriber
        : public transcriber {

        std::vector<std::shared_ptr<transcriber>> layer;

        virtual std::vector<std::shared_ptr<autodiff::op_t>> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;

    };

    struct subsampled_transcriber
        : public transcriber {

        int freq;
        int shift;
        std::shared_ptr<transcriber> base;

        subsampled_transcriber(int freq, int shift, std::shared_ptr<transcriber> base);

        virtual std::vector<std::shared_ptr<autodiff::op_t>> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;
    };

}

#endif
