#ifndef SEQ2SEQ_H
#define SEQ2SEQ_H

#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"

namespace seq2seq {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int encoder_layers);

    struct seq2seq_nn_t {
        std::shared_ptr<autodiff::op_t> pred;
        std::vector<std::shared_ptr<autodiff::op_t>> preds;
        std::vector<std::shared_ptr<autodiff::op_t>> attentions;
    };

    seq2seq_nn_t make_training_nn(
        std::vector<int> labels,
        int label_set_size,
        std::shared_ptr<autodiff::op_t> hidden,
        int nhidden,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree);

}

#endif
