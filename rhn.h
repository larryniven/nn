#ifndef RHN_H
#define RHN_H

#include "la/la.h"
#include "autodiff/autodiff.h"
#include <random>
#include "nn/tensor_tree.h"

namespace rhn {

    std::shared_ptr<tensor_tree::vertex> make_rhn_tensor_tree();

    std::shared_ptr<autodiff::op_t> make_rhn_step_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> input);

    struct rhn_nn_t {
        std::vector<std::shared_ptr<autodiff::op_t>> cell;
        std::vector<std::shared_ptr<autodiff::op_t>> output;
    };

    rhn_nn_t make_rhn_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat, int depth);

    struct rhn_builder {
        rhn_nn_t operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
            int depth) const;
    };

    std::shared_ptr<tensor_tree::vertex> make_bi_rhn_tensor_tree();

    rhn_nn_t make_bi_rhn_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat, int depth);

}

#endif
