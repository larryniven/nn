#ifndef SEQ2SEQ_H
#define SEQ2SEQ_H

#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"

namespace seq2seq {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int encoder_layers);

    struct seq2seq_nn_t {
        std::shared_ptr<autodiff::op_t> pred;
        std::vector<std::shared_ptr<autodiff::op_t>> atts;
    };

    struct attention {
        virtual ~attention();

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> hidden,
            int nhidden,
            int cell_dim) = 0;
    };

    struct bilinear_attention
        : public attention {

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> hidden,
            int nhidden,
            int cell_dim) override;
    };

    struct bilinear_softmax_attention
        : public attention {

        virtual std::shared_ptr<autodiff::op_t> operator()(
            std::shared_ptr<autodiff::op_t> output,
            std::shared_ptr<autodiff::op_t> hidden,
            int nhidden,
            int cell_dim) override;
    };

    seq2seq_nn_t make_training_nn(
        std::vector<int> labels,
        int label_set_size,
        std::shared_ptr<autodiff::op_t> hidden,
        int nhidden,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        attention& att_func);

    std::vector<int> decode(
        std::vector<std::string> const& id_label,
        std::shared_ptr<autodiff::op_t> hidden,
        int nhidden,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        attention& att_func);

}

#endif
