#include "nn/cnn-frame.h"

namespace cnn {

    std::shared_ptr<tensor_tree::vertex> make_cnn_tensor_tree(int layer)
    {
        tensor_tree::vertex root { "nil" };
    
        for (int i = 0; i < layer; ++i) {
            tensor_tree::vertex conv { "nil" };
            conv.children.push_back(tensor_tree::make_tensor("conv weight"));
            conv.children.push_back(tensor_tree::make_tensor("conv bias"));
            root.children.push_back(std::make_shared<tensor_tree::vertex>(conv));
        }
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_densenet_tensor_tree(int layer)
    {
        tensor_tree::vertex root { "nil" };
    
        tensor_tree::vertex conv { "nil" };
        for (int i = 0; i < layer; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                conv.children.push_back(tensor_tree::make_tensor("conv"));
            }
        }
        root.children.push_back(std::make_shared<tensor_tree::vertex>(conv));
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(cnn_t const& config)
    {
        tensor_tree::vertex root { "nil" };
    
        for (int i = 0; i < config.layers.size(); ++i) {
            if (config.layers[i].type == "conv") {
                tensor_tree::vertex conv { "nil" };
                conv.children.push_back(tensor_tree::make_tensor("conv weight"));
                conv.children.push_back(tensor_tree::make_tensor("conv bias"));
                root.children.push_back(std::make_shared<tensor_tree::vertex>(conv));
            } else if (config.layers[i].type == "fc") {
                tensor_tree::vertex fc { "nil" };

                fc.children.push_back(tensor_tree::make_tensor("weight"));
                fc.children.push_back(tensor_tree::make_tensor("bias"));

                root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));
            } else if (config.layers[i].type == "framewise-fc") {
                tensor_tree::vertex fc { "nil" };

                fc.children.push_back(tensor_tree::make_tensor("weight"));
                fc.children.push_back(tensor_tree::make_tensor("bias"));

                root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));
            }
        }
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    cnn_t load_param(std::istream& is)
    {
        std::string line;

        cnn_t result;

        while (std::getline(is, line) && line != "#") {
            if (ebt::startswith(line, "conv")) {
                auto parts = ebt::split(line);
                layer_t ell { "conv" };
                assert(parts.size() == 5);
                ell.data = std::make_shared<std::tuple<int, int, int, int>>(
                    std::make_tuple(std::stoi(parts[1]), std::stoi(parts[2]),
                    std::stoi(parts[4]), std::stoi(parts[4])));
                result.layers.push_back(ell);
            } else if (ebt::startswith(line, "max-pooling")) {
                auto parts = ebt::split(line);
                layer_t ell { "max-pooling" };
                assert(parts.size() == 5);
                ell.data = std::make_shared<std::tuple<int, int, int, int>>(
                    std::make_tuple(std::stoi(parts[1]), std::stoi(parts[2]),
                    std::stoi(parts[3]), std::stoi(parts[4])));
                result.layers.push_back(ell);
            } else if (ebt::startswith(line, "fc")) {
                result.layers.push_back(layer_t { "fc" });
            } else if (ebt::startswith(line, "framewise-fc")) {
                result.layers.push_back(layer_t { "framewise-fc" });
            } else if (ebt::startswith(line, "relu")) {
                result.layers.push_back(layer_t { "relu" });
            } else if (ebt::startswith(line, "logsoftmax")) {
                result.layers.push_back(layer_t { "logsoftmax" });
            } else {
                throw std::logic_error("unable to parse: " + line);
            }
        }

        result.param = make_tensor_tree(result);
        tensor_tree::load_tensor(result.param, is);

        return result;
    }

    void save_param(cnn_t& config, std::ostream& os)
    {
        for (int i = 0; i < config.layers.size(); ++i) {
            auto& ell = config.layers[i];

            if (ell.type == "conv") {
                auto& t = *std::static_pointer_cast<std::tuple<int, int, int, int>>(ell.data);
                os << "conv " << std::get<0>(t) << " " << std::get<1>(t)
                    << " " << std::get<2>(t) << " " << std::get<3>(t) << std::endl;
            } else if (ell.type == "max-pooling") {
                std::tuple<int, int, int, int> t = *std::static_pointer_cast<
                    std::tuple<int, int, int, int>>(ell.data);
                os << "max-pooling " << std::get<0>(t)
                    << " " << std::get<1>(t) << " " << std::get<2>(t)
                    << " " << std::get<3>(t) << std::endl;
            } else if (ell.type == "fc") {
                os << "fc" << std::endl;
            } else if (ell.type == "framewise-fc") {
                os << "framewise-fc" << std::endl;
            } else if (ell.type == "relu") {
                os << "relu" << std::endl;
            } else if (ell.type == "logsoftmax") {
                os << "logsoftmax" << std::endl;
            } else {
                throw std::logic_error("unable to parse: " + ell.type);
            }
        }

        os << "#" << std::endl;

        tensor_tree::save_tensor(config.param, os);
    }

    std::shared_ptr<transcriber>
    make_transcriber(cnn_t const& config, double dropout, std::default_random_engine *gen)
    {
        cnn::multilayer_transcriber multi_trans;

        for (int i = 0; i < config.layers.size(); ++i) {
            auto& ell = config.layers[i];

            if (ell.type == "conv") {
                auto& d = *std::static_pointer_cast<std::tuple<int, int, int, int>>(ell.data);

                auto t = std::make_shared<conv_transcriber>(
                    conv_transcriber { std::get<0>(d), std::get<1>(d),
                    std::get<2>(d), std::get<3>(d) });

                multi_trans.layers.push_back(t);

            } else if (ell.type == "max-pooling") {
                auto& d = *std::static_pointer_cast<std::tuple<int, int, int, int>>(ell.data);

                auto t = std::make_shared<max_pooling_transcriber>(
                    max_pooling_transcriber { std::get<0>(d), std::get<1>(d),
                    std::get<2>(d), std::get<3>(d) });

                multi_trans.layers.push_back(t);

            } else if (ell.type == "fc") {
                auto t = std::make_shared<fc_transcriber>(fc_transcriber{}); 

                multi_trans.layers.push_back(t);

                if (dropout != 0.0) {
                    multi_trans.layers.push_back(std::make_shared<dropout_transcriber>(
                        dropout_transcriber {dropout, *gen}));
                }
            } else if (ell.type == "framewise-fc") {
                auto t = std::make_shared<framewise_fc_transcriber>(framewise_fc_transcriber{}); 

                multi_trans.layers.push_back(t);

                if (dropout != 0.0) {
                    multi_trans.layers.push_back(std::make_shared<dropout_transcriber>(
                        dropout_transcriber {dropout, *gen}));
                }
            } else if (ell.type == "relu") {
                auto t = std::make_shared<relu_transcriber>(relu_transcriber {});
                multi_trans.layers.push_back(t);
            } else if (ell.type == "logsoftmax") {
                auto t = std::make_shared<logsoftmax_transcriber>(logsoftmax_transcriber {});
                multi_trans.layers.push_back(t);
            } else {
                throw std::logic_error("unknown layer type: " + ell.type);
            }
        }

        return std::make_shared<multilayer_transcriber>(multi_trans);

    }

}
