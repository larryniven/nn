#ifndef LSTM_FRAME_H
#define LSTM_FRAME_h

#include <memory>
#include "nn/tensor-tree.h"

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

}

#endif
