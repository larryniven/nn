#include "nn/attention.h"

namespace attention {

    attention_nn_t attend(
        std::shared_ptr<autodiff::op_t> const& hs,
        std::shared_ptr<autodiff::op_t> const& target)
    {
        attention_nn_t att;
    
        att.attention = autodiff::softmax(autodiff::mul(hs, target));
        att.context = autodiff::lmul(att.attention, hs);
    
        return att;
    }

}
