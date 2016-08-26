#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "opt/opt.h"
#include "nn/lstm.h"
#include "nn/pred.h"
#include "nn/nn.h"
#include <random>
#include "nn/tensor_tree.h"
#include "nn/lstm-seg.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gt_batch;

    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> grad;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    int layer;

    double step_size;
    double decay;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

    void learn_sample(
        std::vector<std::vector<double>> const& frames,
        std::string const& label);

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "lstm-seg-li-update",
        "Update LSTM parameters",
        {
            {"param", "", true},
            {"grad", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"uniform-att", "", false},
            {"endpoints", "", false},
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
    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_seg::make_tensor_tree(layer, args);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    std::ifstream grad_ifs { args.at("grad") };
    std::getline(grad_ifs, line);
    grad = lstm_seg::make_tensor_tree(layer, args);
    tensor_tree::load_tensor(grad, grad_ifs);
    grad_ifs.close();

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);
    opt_data = lstm_seg::make_tensor_tree(layer, args);
    tensor_tree::load_tensor(opt_data, opt_data_ifs);
    opt_data_ifs.close();

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }
}

void learning_env::run()
{
    double v1 = tensor_tree::get_matrix(param->children[0]->children[0]->children[0]->children[0])(0, 0);

    if (ebt::in(std::string("decay"), args)) {
        tensor_tree::rmsprop_update(param, grad, opt_data, decay, step_size);
    } else {
        tensor_tree::adagrad_update(param, grad, opt_data, step_size);
    }

    double v2 = tensor_tree::get_matrix(param->children[0]->children[0]->children[0]->children[0])(0, 0);

    std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt_data_ofs << layer << std::endl;
    tensor_tree::save_tensor(opt_data, opt_data_ofs);
    opt_data_ofs.close();
}

