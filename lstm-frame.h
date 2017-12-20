#ifndef LSTM_FRAME_H
#define LSTM_FRAME_H

#include <memory>
#include "nn/tensor-tree.h"
#include "nn/lstm.h"
#include <random>

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

    std::shared_ptr<tensor_tree::vertex> make_dyer_tensor_tree(int layer);

    std::shared_ptr<lstm::transcriber>
    make_transcriber(
        std::shared_ptr<tensor_tree::vertex> param,
        double dropout,
        std::default_random_engine *gen,
        bool pyramid);

    std::shared_ptr<lstm::transcriber>
    make_dyer_transcriber(
        std::shared_ptr<tensor_tree::vertex> param,
        double dropout,
        std::default_random_engine *gen,
        bool pyramid);

}

#endif
