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

    // step transcriber

    struct step_transcriber {

        virtual ~step_transcriber();

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) = 0;

    };

    struct lstm_step_transcriber
        : public step_transcriber {

        std::shared_ptr<autodiff::op_t> cell;
        std::shared_ptr<autodiff::op_t> output;

        lstm_step_transcriber();

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input);

    };

    struct input_dropout_transcriber
        : public step_transcriber {

        std::default_random_engine& gen;
        double prob;
        std::shared_ptr<step_transcriber> base;

        input_dropout_transcriber(std::default_random_engine& gen, double prob,
            std::shared_ptr<step_transcriber> base);

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input);
    };

    struct output_dropout_transcriber
        : public step_transcriber {

        std::default_random_engine& gen;
        double prob;
        std::shared_ptr<step_transcriber> base;

        output_dropout_transcriber(std::default_random_engine& gen, double prob,
            std::shared_ptr<step_transcriber> base);

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input);
    };

    struct dyer_lstm_step_transcriber
        : public step_transcriber {

        std::shared_ptr<autodiff::op_t> cell;
        std::shared_ptr<autodiff::op_t> output;

        dyer_lstm_step_transcriber();

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;

    };

    struct lstm_multistep_transcriber
        : public step_transcriber {

        std::vector<std::shared_ptr<step_transcriber>> steps;

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input);

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

        std::shared_ptr<step_transcriber> step;

        lstm_transcriber(std::shared_ptr<step_transcriber> step);

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

    struct logsoftmax_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;

        logsoftmax_transcriber(std::shared_ptr<transcriber> base);

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

    struct hypercolumn_transcriber
        : public transcriber {

        std::vector<std::shared_ptr<transcriber>> layer;

        virtual std::vector<std::shared_ptr<autodiff::op_t>> operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const override;

    };

}

#endif
