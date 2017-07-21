#ifndef CNN_FRAME_H
#define CNN_FRAME_H

#include "nn/tensor-tree.h"
#include "nn/cnn.h"

namespace cnn {

    struct layer_t {
        std::string type;
        std::shared_ptr<void> data;
    };

    struct cnn_t {
        std::vector<layer_t> layers;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    std::shared_ptr<tensor_tree::vertex> make_cnn_tensor_tree(int layer);

    std::shared_ptr<tensor_tree::vertex> make_densenet_tensor_tree(int layer);

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(cnn_t const& config);

    cnn_t load_param(std::istream& is);
    void save_param(cnn_t& param, std::ostream& os);

    std::shared_ptr<transcriber>
    make_transcriber(cnn_t const& cnn_config, double dropout, std::default_random_engine *gen);

}

#endif
