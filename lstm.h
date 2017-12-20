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

    // tanscriber

    struct trans_seq_t {
        int nframes;
        int batch_size;
        int dim;
        std::shared_ptr<autodiff::op_t> feat;
        std::shared_ptr<autodiff::op_t> mask;
    };

    trans_seq_t make_trans_seq(std::shared_ptr<autodiff::op_t> t);

    struct transcriber {
        virtual ~transcriber();

        virtual bool require_param() const override;

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const = 0;
    };

    struct lstm_transcriber
        : public transcriber {

        int cell_dim;
        bool reverse;

        lstm_transcriber(int cell_dim, bool reverse = false);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;

    };

    struct dyer_lstm_transcriber
        : public transcriber {

        int cell_dim;
        bool reverse;

        dyer_lstm_transcriber(int cell_dim, bool reverse = false);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;

    };

    struct input_dropout_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;
        double prob;
        std::default_random_engine& gen;

        input_dropout_transcriber(std::shared_ptr<transcriber> base,
            double prob, std::default_random_engine& gen);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;

    };

    struct output_dropout_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;
        double prob;
        std::default_random_engine& gen;

        output_dropout_transcriber(std::shared_ptr<transcriber> base,
            double prob, std::default_random_engine& gen);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

    struct bi_transcriber
        : public transcriber {

        int output_dim;

        std::shared_ptr<transcriber> f_base;
        std::shared_ptr<transcriber> b_base;

        bi_transcriber(int output_dim, std::shared_ptr<transcriber> f_base,
            std::shared_ptr<transcriber> b_base);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

    struct layered_transcriber
        : public transcriber {

        std::vector<std::shared_ptr<transcriber>> layer;

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

    struct fc_transcriber
        : public transcriber {

        int output_dim;

        fc_transcriber(int output_dim);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

    struct logsoftmax_transcriber
        : public transcriber {

        virtual bool require_param() const override;

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

    struct res_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;

        res_transcriber(std::shared_ptr<transcriber> base);

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

    struct subsampled_transcriber
        : public transcriber {

        int freq;
        int shift;

        subsampled_transcriber(int freq, int shift);

        virtual bool require_param() const override;

        virtual trans_seq_t operator()(
            std::shared_ptr<tensor_tree::vertex> var_tree,
            trans_seq_t const& seq) const override;
    };

}

#endif
