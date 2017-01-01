#include "nn/cnn.h"

namespace cnn {

    std::shared_ptr<tensor_tree::vertex> make_cnn_tensor_tree(int layer)
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };
    
        for (int i = 0; i < layer; ++i) {
            tensor_tree::vertex conv { tensor_tree::tensor_t::nil };
            conv.children.push_back(tensor_tree::make_tensor("conv weight"));
            conv.children.push_back(tensor_tree::make_tensor("conv bias"));
            root.children.push_back(std::make_shared<tensor_tree::vertex>(conv));
        }
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_densenet_tensor_tree(int layer)
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };
    
        tensor_tree::vertex conv { tensor_tree::tensor_t::nil };
        for (int i = 0; i < layer; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                conv.children.push_back(tensor_tree::make_tensor("conv"));
            }
        }
        root.children.push_back(std::make_shared<tensor_tree::vertex>(conv));
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<autodiff::op_t>
    cnn_transcriber::operator()(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        auto k = autodiff::corr_linearize(input,
            tensor_tree::get_var(var_tree->children[0]));

        auto z = autodiff::mul(k, tensor_tree::get_var(var_tree->children[0]));

        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), z);

        return autodiff::relu(autodiff::add(z, b));
    }

    dropout_transcriber::dropout_transcriber(std::shared_ptr<transcriber> base,
        double prob, std::default_random_engine& gen)
        : base(base), prob(prob), gen(gen)
    {}

    std::shared_ptr<autodiff::op_t>
    dropout_transcriber::operator()(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        input = autodiff::emul(input, autodiff::dropout_mask(input, prob, gen));
        return (*base)(input, var_tree);
    }

    std::shared_ptr<autodiff::op_t>
    multilayer_transcriber::operator()(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        std::shared_ptr<autodiff::op_t> feat = input;

        for (int i = 0; i < layers.size(); ++i) {
            feat = (*layers[i])(feat, var_tree->children[i]);
        }

        return feat;
    }

    densenet_transcriber::densenet_transcriber(int layer)
        : layer(layer)
    {}

    std::shared_ptr<autodiff::op_t>
    densenet_transcriber::operator()(std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> layers;

        int ell = 0;

        std::shared_ptr<autodiff::op_t> feat = input;

        for (int i = 0; i < layer; ++i) {
            auto t = autodiff::corr_linearize(feat,
                tensor_tree::get_var(var_tree->children[ell + i]));
            layers.push_back(t);

            std::vector<std::shared_ptr<autodiff::op_t>> muls;
            for (int k = 0; k < i + 1; ++k) {
                muls.push_back(autodiff::mul(layers[k],
                    tensor_tree::get_var(var_tree->children[ell + k])));
            }
            feat = autodiff::relu(autodiff::add(muls));

            ell += i + 1;
        }

        return input;
    }

}
