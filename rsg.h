#ifndef RSG_H
#define RSG_H

#include "nn/tensor-tree.h"
#include "autodiff/autodiff.h"

namespace rsg {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree();

    std::vector<std::shared_ptr<autodiff::op_t>> make_training_nn(
        std::shared_ptr<autodiff::op_t> init_cell,
        std::vector<std::shared_ptr<autodiff::op_t>> const& gt_seq,
        std::shared_ptr<tensor_tree::vertex> var_tree);

    std::vector<std::shared_ptr<autodiff::op_t>> make_nn(
        std::shared_ptr<autodiff::op_t> init_cell,
        std::shared_ptr<autodiff::op_t> init_input,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        int steps);

}

#endif
