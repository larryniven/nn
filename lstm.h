#ifndef LSTM_H
#define LSTM_H

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

    lstm_step_nn_t make_lstm_step_nn(
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> prev_output,
        std::shared_ptr<autodiff::op_t> prev_cell,
        std::shared_ptr<autodiff::op_t> output_weight,
        std::shared_ptr<autodiff::op_t> cell2i,
        std::shared_ptr<autodiff::op_t> cell2f,
        std::shared_ptr<autodiff::op_t> cell2o,
        std::shared_ptr<autodiff::op_t> cell_mask,
        std::shared_ptr<autodiff::op_t> output_storage,
        int batch_size,
        int cell_dim);

    lstm_step_nn_t make_dyer_lstm_step_nn(
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> prev_output,
        std::shared_ptr<autodiff::op_t> prev_cell,
        std::shared_ptr<autodiff::op_t> output_weight,
        std::shared_ptr<autodiff::op_t> cell2i,
        std::shared_ptr<autodiff::op_t> cell2o,
        std::shared_ptr<autodiff::op_t> cell_mask,
        std::shared_ptr<autodiff::op_t> output_storage,
        int batch_size,
        int cell_dim);

#if 0
    lstm_step_nn_t make_dyer_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> prev_cell,
        std::shared_ptr<autodiff::op_t> prev_output,
        std::shared_ptr<autodiff::op_t> input_h,
        std::shared_ptr<autodiff::op_t> input_i,
        std::shared_ptr<autodiff::op_t> input_o,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> cell_mask = nullptr);
#endif

    // tanscriber

    struct transcriber {
        virtual ~transcriber();

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const = 0;
    };

    struct lstm_transcriber
        : public transcriber {

        bool reverse;

        lstm_transcriber(bool reverse = false);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    std::vector<std::shared_ptr<autodiff::op_t>> split_rows(
        std::shared_ptr<autodiff::op_t> t);

    struct dyer_lstm_transcriber
        : public transcriber {

        bool reverse;

        dyer_lstm_transcriber(bool reverse = false);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    struct input_dropout_transcriber
        : public transcriber {

        mutable std::shared_ptr<autodiff::op_t> debug;

        std::shared_ptr<transcriber> base;
        double prob;
        std::default_random_engine& gen;

        input_dropout_transcriber(std::shared_ptr<transcriber> base,
            double prob, std::default_random_engine& gen);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    struct output_dropout_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;
        double prob;
        std::default_random_engine& gen;

        output_dropout_transcriber(std::shared_ptr<transcriber> base,
            double prob, std::default_random_engine& gen);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    struct bi_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> f_base;
        std::shared_ptr<transcriber> b_base;

        bi_transcriber(std::shared_ptr<transcriber> f_base,
            std::shared_ptr<transcriber> b_base);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;
    };

    struct layered_transcriber
        : public transcriber {

        std::vector<std::shared_ptr<transcriber>> layer;

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    struct logsoftmax_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;

        logsoftmax_transcriber(std::shared_ptr<transcriber> base);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    struct res_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;

        res_transcriber(std::shared_ptr<transcriber> base);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;

    };

    struct subsampled_transcriber
        : public transcriber {

        int freq;
        int shift;
        std::shared_ptr<transcriber> base;

        subsampled_transcriber(int freq, int shift, std::shared_ptr<transcriber> base);

        virtual
        std::pair<std::shared_ptr<autodiff::op_t>,
            std::shared_ptr<autodiff::op_t>>
        operator()(
            int nframes,
            int batch_size,
            int cell_dim,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> const& feat,
            std::shared_ptr<autodiff::op_t> const& mask = nullptr) const override;
    };

}

#endif
