#include "nn/lstm-frame.h"
#include "nn/lstm.h"
#include "nn/pred.h"

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        tensor_tree::vertex result { tensor_tree::tensor_t::nil };
    
        result.children.push_back(lstm::make_stacked_bi_lstm_tensor_tree(layer));
        result.children.push_back(nn::make_pred_tensor_tree());
    
        return std::make_shared<tensor_tree::vertex>(result);
    }

}
