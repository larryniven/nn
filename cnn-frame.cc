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

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int conv_layer, int fc_layer)
    {
        std::shared_ptr<tensor_tree::vertex> root = make_cnn_tensor_tree(conv_layer);
    
        for (int i = 0; i < fc_layer; ++i) {
            tensor_tree::vertex t { "nil" };

            t.children.push_back(tensor_tree::make_tensor("softmax weight"));
            t.children.push_back(tensor_tree::make_tensor("softmax bias"));

            root->children.push_back(std::make_shared<tensor_tree::vertex>(t));
        }
    
        tensor_tree::vertex t { "nil" };

        t.children.push_back(tensor_tree::make_tensor("softmax weight"));
        t.children.push_back(tensor_tree::make_tensor("softmax bias"));

        root->children.push_back(std::make_shared<tensor_tree::vertex>(t));

        return root;
    }

    cnn_t load_param(std::istream& is)
    {
        std::string line;

        if (!std::getline(is, line)) {
            std::cout << "fail to parse the number of conv layers" << std::endl;
            exit(1);
        }

        cnn_t result;

        result.conv_layer = std::stoi(line);

        for (int i = 0; i < result.conv_layer; ++i) {
            if (!std::getline(is, line)) {
                std::cout << "fail to parse dilation parameters" << std::endl;
                exit(1);
            }

            std::vector<std::string> parts = ebt::split(line);

            assert(parts.size() == 2);

            result.dilation.push_back(std::make_pair(std::stoi(parts[0]), std::stoi(parts[1])));
        }

        if (!std::getline(is, line)) {
            std::cout << "fail to parse the number of fc layers" << std::endl;
            exit(1);
        }

        result.fc_layer = std::stoi(line);

        result.param = make_tensor_tree(result.conv_layer, result.fc_layer);
        tensor_tree::load_tensor(result.param, is);

        return result;
    }

    void save_param(cnn_t& config, std::ostream& os)
    {
        os << config.conv_layer << std::endl;

        for (int i = 0; i < config.conv_layer; ++i) {
            os << config.dilation[i].first << " " << config.dilation[i].second << std::endl;
        }

        os << config.fc_layer << std::endl;

        tensor_tree::save_tensor(config.param, os);
    }

    std::shared_ptr<transcriber>
    make_transcriber(cnn_t const& cnn_config, double dropout, std::default_random_engine *gen)
    {
        cnn::multilayer_transcriber multi_trans;

        for (int i = 0; i < cnn_config.conv_layer; ++i) {
            auto t = std::make_shared<cnn_transcriber>(
                cnn_transcriber { cnn_config.dilation[i].first, cnn_config.dilation[i].second });

            multi_trans.layers.push_back(t);
        }

        for (int i = 0; i < cnn_config.fc_layer; ++i) {
            auto t = std::make_shared<fc_transcriber>(fc_transcriber{}); 

            if (dropout == 0.0) {
                multi_trans.layers.push_back(t);
            } else {
                multi_trans.layers.push_back(std::make_shared<dropout_transcriber>(
                    dropout_transcriber {t, dropout, *gen}));
            }
        }

        return std::make_shared<multilayer_transcriber>(multi_trans);

    }

}
