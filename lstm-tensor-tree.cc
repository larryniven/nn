#include "lstm-tensor-tree.h"

namespace lstm {

    lstm_tensor_tree_factory::~lstm_tensor_tree_factory()
    {}

    std::shared_ptr<tensor_tree::vertex> lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

        // 0
        root.children.push_back(tensor_tree::make_matrix("input -> hidden"));
        root.children.push_back(tensor_tree::make_matrix("output -> hidden"));
        root.children.push_back(tensor_tree::make_vector("hidden bias"));

        // 3
        root.children.push_back(tensor_tree::make_matrix("input -> input gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> input gate"));
        root.children.push_back(tensor_tree::make_vector("input gate peep"));
        root.children.push_back(tensor_tree::make_vector("input gate bias"));

        // 7
        root.children.push_back(tensor_tree::make_matrix("input -> output gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> output gate"));
        root.children.push_back(tensor_tree::make_vector("output gate peep"));
        root.children.push_back(tensor_tree::make_vector("output gate bias"));

        // 11
        root.children.push_back(tensor_tree::make_matrix("input -> forget gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> forget gate"));
        root.children.push_back(tensor_tree::make_vector("forget gate peep"));
        root.children.push_back(tensor_tree::make_vector("forget gate bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> dyer_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

        // 0
        root.children.push_back(tensor_tree::make_matrix("input -> hidden"));
        root.children.push_back(tensor_tree::make_matrix("output -> hidden"));
        root.children.push_back(tensor_tree::make_vector("hidden bias"));

        // 3
        root.children.push_back(tensor_tree::make_matrix("input -> input gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> input gate"));
        root.children.push_back(tensor_tree::make_matrix("input gate peep"));
        root.children.push_back(tensor_tree::make_vector("input gate bias"));

        // 7
        root.children.push_back(tensor_tree::make_matrix("input -> output gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> output gate"));
        root.children.push_back(tensor_tree::make_matrix("output gate peep"));
        root.children.push_back(tensor_tree::make_vector("output gate bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    multilayer_lstm_tensor_tree_factory::multilayer_lstm_tensor_tree_factory(
        std::shared_ptr<lstm_tensor_tree_factory> base_fac, int layer)
        : base_fac(base_fac), layer(layer)
    {}

    std::shared_ptr<tensor_tree::vertex> multilayer_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

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

        root.children.push_back(tensor_tree::make_matrix("forward output weight"));
        root.children.push_back(tensor_tree::make_matrix("backward output weight"));
        root.children.push_back(tensor_tree::make_vector("output bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_bi_lstm_tensor_tree()
    {
        bi_lstm_tensor_tree_factory f;
        return f();
    }

    stacked_bi_lstm_tensor_tree_factory::stacked_bi_lstm_tensor_tree_factory(int layer)
        : layer(layer)
        , base_fac(std::make_shared<bi_lstm_tensor_tree_factory>(bi_lstm_tensor_tree_factory{}))
    {}

    stacked_bi_lstm_tensor_tree_factory::stacked_bi_lstm_tensor_tree_factory(
        int layer,
        std::shared_ptr<bi_lstm_tensor_tree_factory> base_fac)
        : layer(layer), base_fac(base_fac)
    {}

    stacked_bi_lstm_tensor_tree_factory::~stacked_bi_lstm_tensor_tree_factory()
    {}

    std::shared_ptr<tensor_tree::vertex> stacked_bi_lstm_tensor_tree_factory::operator()() const
    {
        tensor_tree::vertex root;

        for (int i = 0; i < layer; ++i) {
            root.children.push_back((*base_fac)());
        }

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_stacked_bi_lstm_tensor_tree(int layer)
    {
        stacked_bi_lstm_tensor_tree_factory f(layer);
        return f();
    }

}
