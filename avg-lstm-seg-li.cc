#include "ebt/ebt.h"
#include <fstream>
#include "nn/tensor_tree.h"
#include "nn/lstm-seg.h"

struct learning_env {

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "avg-lstm-seg-li",
        "Average LSTM parameters",
        {
            {"params", "", true},
            {"output", "", true},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
}

void learning_env::run()
{
    std::shared_ptr<tensor_tree::vertex> param_avg;

    int layer;

    std::vector<std::string> param_files = ebt::split(args.at("params"), ",");

    for (int i = 0; i < param_files.size(); ++i) {
        std::string line;

        if (i == 0) {
            std::ifstream ifs { param_files[i] };
            std::getline(ifs, line);
            layer = std::stoi(line);
            param_avg = lstm_seg::make_tensor_tree(layer, args);
            tensor_tree::load_tensor(param_avg, ifs);
            ifs.close();
        } else {
            std::shared_ptr<tensor_tree::vertex> param = lstm_seg::make_tensor_tree(layer, args);

            std::ifstream ifs { param_files[i] };
            std::getline(ifs, line);
            tensor_tree::load_tensor(param, ifs);
            ifs.close();

            tensor_tree::iadd(param_avg, param);
        }
    }

    tensor_tree::imul(param_avg, 1.0 / param_files.size());

    std::ofstream ofs { args.at("output") };
    ofs << layer << std::endl;
    tensor_tree::save_tensor(param_avg, ofs);
    ofs.close();

}

