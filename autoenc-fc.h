#ifndef NN_AUTOENC_H
#define NN_AUTOENC_H

#include "nn/tensor-tree.h"

namespace autoenc {

    std::shared_ptr<tensor_tree::vertex> make_symmetric_ae_tensor_tree();

    std::shared_ptr<autodiff::op_t> make_symmetric_ae(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        double input_dropout, double hidden_dropout,
        std::default_random_engine *gen);

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

    std::shared_ptr<autodiff::op_t> make_nn(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        double input_dropout, double hidden_dropout,
        std::default_random_engine& gen);

    std::shared_ptr<autodiff::op_t> make_wta_nn(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        int k);

}

#endif
