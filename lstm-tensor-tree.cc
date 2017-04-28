#include "lstm-tensor-tree.h"

namespace lstm {

    lstm_tensor_tree_factory::~lstm_tensor_tree_factory()
    {}

    std::shared_ptr<tensor_tree::vertex> lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { "nil" };

        // 0
        root.children.push_back(tensor_tree::make_tensor("input -> hidden"));
        root.children.push_back(tensor_tree::make_tensor("output -> hidden"));
        root.children.push_back(tensor_tree::make_tensor("hidden bias"));

        // 3
        root.children.push_back(tensor_tree::make_tensor("input -> input gate"));
        root.children.push_back(tensor_tree::make_tensor("output -> input gate"));
        root.children.push_back(tensor_tree::make_tensor("input gate peep"));
        root.children.push_back(tensor_tree::make_tensor("input gate bias"));

        // 7
        root.children.push_back(tensor_tree::make_tensor("input -> output gate"));
        root.children.push_back(tensor_tree::make_tensor("output -> output gate"));
        root.children.push_back(tensor_tree::make_tensor("output gate peep"));
        root.children.push_back(tensor_tree::make_tensor("output gate bias"));

        // 11
        root.children.push_back(tensor_tree::make_tensor("input -> forget gate"));
        root.children.push_back(tensor_tree::make_tensor("output -> forget gate"));
        root.children.push_back(tensor_tree::make_tensor("forget gate peep"));
        root.children.push_back(tensor_tree::make_tensor("forget gate bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> dyer_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { "nil" };

        // 0
        root.children.push_back(tensor_tree::make_tensor("input -> hidden"));
        root.children.push_back(tensor_tree::make_tensor("output -> hidden"));
        root.children.push_back(tensor_tree::make_tensor("hidden bias"));

        // 3
        root.children.push_back(tensor_tree::make_tensor("input -> input gate"));
        root.children.push_back(tensor_tree::make_tensor("output -> input gate"));
        root.children.push_back(tensor_tree::make_tensor("input gate peep"));
        root.children.push_back(tensor_tree::make_tensor("input gate bias"));

        // 7
        root.children.push_back(tensor_tree::make_tensor("input -> output gate"));
        root.children.push_back(tensor_tree::make_tensor("output -> output gate"));
        root.children.push_back(tensor_tree::make_tensor("output gate peep"));
        root.children.push_back(tensor_tree::make_tensor("output gate bias"));

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
        tensor_tree::vertex root;

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
