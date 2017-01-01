#ifndef CNN_H
#define CNN_H

#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"

namespace cnn {

    std::shared_ptr<tensor_tree::vertex> make_cnn_tensor_tree(int layer);

    std::shared_ptr<tensor_tree::vertex> make_densenet_tensor_tree(int layer);

    struct transcriber {
        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<autodiff::op_t> input,
            std::shared_ptr<tensor_tree::vertex> var_tree) = 0;
    };

    struct cnn_transcriber
        : public transcriber {

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<autodiff::op_t> input,
            std::shared_ptr<tensor_tree::vertex> var_tree) override;
    };

    struct dropout_transcriber
        : public transcriber {

        std::shared_ptr<transcriber> base;
        double prob;
        std::default_random_engine& gen;

        dropout_transcriber(std::shared_ptr<transcriber> base,
            double prob, std::default_random_engine& gen);

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<autodiff::op_t> input,
            std::shared_ptr<tensor_tree::vertex> var_tree) override;
    };

    struct multilayer_transcriber
        : public transcriber {

        std::vector<std::shared_ptr<transcriber>> layers;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<autodiff::op_t> input,
            std::shared_ptr<tensor_tree::vertex> var_tree) override;
    };

    struct densenet_transcriber
        : public transcriber {

        int layer;

        densenet_transcriber(int layer);

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<autodiff::op_t> input,
            std::shared_ptr<tensor_tree::vertex> var_tree) override;
    };

}

#endif
