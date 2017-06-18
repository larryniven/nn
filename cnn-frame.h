#ifndef CNN_FRAME_H
#define CNN_FRAME_H

#include "nn/tensor-tree.h"
#include "nn/cnn.h"

namespace cnn {

    std::shared_ptr<tensor_tree::vertex> make_cnn_tensor_tree(int layer);

    std::shared_ptr<tensor_tree::vertex> make_densenet_tensor_tree(int layer);

    struct cnn_t {
        int conv_layer;
        int fc_layer;
        std::vector<std::pair<int, int>> dilation;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int conv_layer, int fc_layer);

    cnn_t load_param(std::istream& is);
    void save_param(cnn_t& param, std::ostream& os);

    std::shared_ptr<transcriber>
    make_transcriber(cnn_t const& cnn_config, double dropout, std::default_random_engine *gen);

}

#endif
