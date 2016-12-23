#include "nn/lstm-frame.h"
#include "nn/lstm.h"
#include "nn/nn.h"
#include "nn/lstm-tensor-tree.h"

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        tensor_tree::vertex result { tensor_tree::tensor_t::nil };

        lstm::multilayer_lstm_tensor_tree_factory factory {
            std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
            lstm::bi_lstm_tensor_tree_factory {
                std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                    lstm::dyer_lstm_tensor_tree_factory{})
            }),
            layer
        };

        result.children.push_back(factory());
        result.children.push_back(tensor_tree::make_tensor("softmax weight"));
        result.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(result);
    }

}
