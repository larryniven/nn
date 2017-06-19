#ifndef LSTM_FRAME_H
#define LSTM_FRAME_h

#include <memory>
#include "nn/tensor-tree.h"
#include "nn/lstm.h"
#include <random>

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

    std::shared_ptr<tensor_tree::vertex> make_uni_tensor_tree(int layer);

    std::shared_ptr<lstm::transcriber>
    make_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen);

    std::shared_ptr<lstm::transcriber>
    make_res_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen);

    std::shared_ptr<lstm::transcriber>
    make_uni_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen);

    std::shared_ptr<lstm::transcriber>
    make_pyramid_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen);

}

#endif
