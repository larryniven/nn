#ifndef CNN_H
#define CNN_H

#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"

namespace cnn {

    struct transcriber {
        virtual ~transcriber();

        virtual bool require_param() const;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) = 0;
    };

    struct conv_transcriber
        : public transcriber {

        int p1;
        int p2;
        int d1;
        int d2;

        conv_transcriber();
        conv_transcriber(int p1, int p2, int d1, int d2);

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct max_pooling_transcriber
        : public transcriber {

        int dim1;
        int dim2;
        int stride1;
        int stride2;

        max_pooling_transcriber(int dim1, int dim2);
        max_pooling_transcriber(int dim1, int dim2, int stride1, int stride2);

        virtual bool require_param() const override;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct fc_transcriber
        : public transcriber {

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct framewise_fc_transcriber
        : public transcriber {

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct relu_transcriber
        : public transcriber {

        virtual bool require_param() const override;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct dropout_transcriber
        : public transcriber {

        double prob;
        std::default_random_engine& gen;

        dropout_transcriber(double prob, std::default_random_engine& gen);

        virtual bool require_param() const override;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct multilayer_transcriber
        : public transcriber {

        std::vector<std::shared_ptr<transcriber>> layers;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

    struct logsoftmax_transcriber
        : public transcriber {

        virtual bool require_param() const override;

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;

    };

    struct densenet_transcriber
        : public transcriber {

        int transcriber;

        densenet_transcriber(int transcriber);

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;

    };

    std::vector<std::shared_ptr<autodiff::op_t>> ifo_pooling(
        std::vector<std::shared_ptr<autodiff::op_t>> input,
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate,
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate,
        std::vector<std::shared_ptr<autodiff::op_t>> output_gate);

    std::vector<std::shared_ptr<autodiff::op_t>> conv_ifo_pooling(
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input_gate,
        std::shared_ptr<autodiff::op_t> forget_gate,
        std::shared_ptr<autodiff::op_t> output_gate,
        int size);

    std::vector<std::shared_ptr<autodiff::op_t>> conv_fo_pooling(
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> forget_gate,
        std::shared_ptr<autodiff::op_t> output_gate,
        int size);

    struct conv_fo_pooling_transcriber
        : public transcriber {

        unsigned int rows;
        unsigned int cols;

        std::shared_ptr<transcriber> input_conv;
        std::shared_ptr<transcriber> forget_gate_conv;
        std::shared_ptr<transcriber> output_gate_conv;

        conv_fo_pooling_transcriber(
            unsigned int rows, unsigned int cols,
            std::shared_ptr<transcriber> input,
            std::shared_ptr<transcriber> forget_gate,
            std::shared_ptr<transcriber> output_gate);

        virtual std::shared_ptr<autodiff::op_t>
        operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
            std::shared_ptr<autodiff::op_t> input) override;
    };

}

#endif
