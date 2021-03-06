#include "lstm-tensor-tree.h"

namespace lstm {

    lstm_tensor_tree_factory::~lstm_tensor_tree_factory()
    {}

    std::shared_ptr<tensor_tree::vertex> lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { "nil" };

        root.children.push_back(tensor_tree::make_tensor("input -> all"));
        root.children.push_back(tensor_tree::make_tensor("bias"));
        root.children.push_back(tensor_tree::make_tensor("output -> all"));
        root.children.push_back(tensor_tree::make_tensor("cell -> input gate"));
        root.children.push_back(tensor_tree::make_tensor("cell -> forget gate"));
        root.children.push_back(tensor_tree::make_tensor("cell -> output gate"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> dyer_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { "nil" };

        root.children.push_back(tensor_tree::make_tensor("input -> all"));
        root.children.push_back(tensor_tree::make_tensor("bias"));
        root.children.push_back(tensor_tree::make_tensor("output -> all"));
        root.children.push_back(tensor_tree::make_tensor("cell -> input gate"));
        root.children.push_back(tensor_tree::make_tensor("cell -> output gate"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    multilayer_lstm_tensor_tree_factory::multilayer_lstm_tensor_tree_factory(
        std::shared_ptr<lstm_tensor_tree_factory> base_fac, int layer)
        : base_fac(base_fac), layer(layer)
    {}

    std::shared_ptr<tensor_tree::vertex> multilayer_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { "nil" };

        for (int i = 0; i < layer; ++i) {
            root.children.push_back((*base_fac)());
        }

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree()
    {
        lstm_tensor_tree_factory f;
        return f();
    }

    bi_lstm_tensor_tree_factory::bi_lstm_tensor_tree_factory()
        : base_fac(std::make_shared<lstm_tensor_tree_factory>(lstm_tensor_tree_factory{}))
    {}

    bi_lstm_tensor_tree_factory::bi_lstm_tensor_tree_factory(
        std::shared_ptr<lstm_tensor_tree_factory> base_fac)
        : base_fac(base_fac)
    {}

    bi_lstm_tensor_tree_factory::~bi_lstm_tensor_tree_factory()
    {}

    std::shared_ptr<tensor_tree::vertex> bi_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { "nil" };

        root.children.push_back((*base_fac)());
        root.children.push_back((*base_fac)());

        root.children.push_back(tensor_tree::make_tensor("forward output weight"));
        root.children.push_back(tensor_tree::make_tensor("backward output weight"));
        root.children.push_back(tensor_tree::make_tensor("output bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_bi_lstm_tensor_tree()
    {
        bi_lstm_tensor_tree_factory f;
        return f();
    }

    std::shared_ptr<tensor_tree::vertex> make_stacked_bi_lstm_tensor_tree(int layer)
    {
        multilayer_lstm_tensor_tree_factory factory { 
            std::make_shared<bi_lstm_tensor_tree_factory>(
                bi_lstm_tensor_tree_factory {
                    std::make_shared<lstm_tensor_tree_factory>(lstm_tensor_tree_factory {})
                }),
            layer
        };
        return factory();
    }

}
