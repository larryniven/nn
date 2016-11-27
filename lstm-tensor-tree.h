#ifndef LSTM_TENSOR_TREE_H
#define LSTM_TENSOR_TREE_H

#include <memory>
#include "nn/tensor-tree.h"

namespace lstm {

    struct lstm_tensor_tree_factory {

        virtual ~lstm_tensor_tree_factory();

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    struct dyer_lstm_tensor_tree_factory
        : public lstm_tensor_tree_factory {

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    struct multilayer_lstm_tensor_tree_factory
        : public lstm_tensor_tree_factory {

        std::shared_ptr<lstm_tensor_tree_factory> base_fac;
        int layer;

        multilayer_lstm_tensor_tree_factory(std::shared_ptr<lstm_tensor_tree_factory> base_fac, int layer);

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree();

    struct bi_lstm_tensor_tree_factory {

        std::shared_ptr<lstm_tensor_tree_factory> base_fac;

        bi_lstm_tensor_tree_factory();
        bi_lstm_tensor_tree_factory(std::shared_ptr<lstm_tensor_tree_factory> base_fac);

        virtual ~bi_lstm_tensor_tree_factory();

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    std::shared_ptr<tensor_tree::vertex> make_bi_lstm_tensor_tree();

    struct stacked_bi_lstm_tensor_tree_factory {

        int layer;
        std::shared_ptr<bi_lstm_tensor_tree_factory> base_fac;

        stacked_bi_lstm_tensor_tree_factory(int layer);
        stacked_bi_lstm_tensor_tree_factory(int layer, std::shared_ptr<bi_lstm_tensor_tree_factory> base_fac);

        virtual ~stacked_bi_lstm_tensor_tree_factory();

        virtual std::shared_ptr<tensor_tree::vertex> operator()() const;

    };

    std::shared_ptr<tensor_tree::vertex> make_stacked_bi_lstm_tensor_tree(int layer);

}

#endif
