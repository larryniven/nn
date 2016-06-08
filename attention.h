#ifndef ATTENTION_H
#define ATTENTION_H

#include "autodiff/autodiff.h"

namespace attention {

    struct attention_nn_t {
        std::shared_ptr<autodiff::op_t> attention;
        std::shared_ptr<autodiff::op_t> context;
    };
    
    attention_nn_t attend(
        std::shared_ptr<autodiff::op_t> const& inputs,
        std::shared_ptr<autodiff::op_t> const& target);

}

#endif
