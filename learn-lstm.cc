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

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    std::shared_ptr<tensor_tree::vertex> pred_param;
    std::shared_ptr<tensor_tree::vertex> pred_opt_data;

    std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
    std::shared_ptr<tensor_tree::vertex> pred_var_tree;

    int layer;

    lstm::stacked_bi_lstm_nn_t nn;
    rnn::pred_nn_t pred_nn;

    double step_size;
    double decay;

    double dropout;
    int dropout_seed;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    double clip;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-lstm",
        "Train a LSTM frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"clip", "", false},
            {"label", "", true},
            {"ignored", "", false},
            {"dropout", "", false},
            {"dropout-seed", "", false}
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
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);

    param = lstm::make_stacked_bi_lstm_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    pred_param = nn::make_pred_tensor_tree();
    tensor_tree::load_tensor(pred_param, param_ifs);
    param_ifs.close();

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);

    opt_data = lstm::make_stacked_bi_lstm_tensor_tree(layer);
    tensor_tree::load_tensor(opt_data, opt_data_ifs);
    pred_opt_data = nn::make_pred_tensor_tree();
    tensor_tree::load_tensor(pred_opt_data, opt_data_ifs);
    opt_data_ifs.close();

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

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

    clip = std::numeric_limits<double>::infinity();
    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    std::vector<std::string> label_vec = speech::load_label_set(args.at("label"));
    for (int i = 0; i < label_vec.size(); ++i) {
        label_id[label_vec[i]] = i;
    }

    if (ebt::in(std::string("ignored"), args)) {
        auto parts = ebt::split(args.at("ignored"), ",");
        ignored.insert(parts.begin(), parts.end());
    }

    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("dropout-seed"), args)) {
        dropout_seed = std::stoi(args.at("dropout-seed"));
    }
}

void learning_env::run()
{
    int i = 1;

    std::default_random_engine gen { dropout_seed };

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

	std::vector<std::string> labels;

	labels = speech::load_label_batch(label_batch);

        if (!frame_batch || !label_batch) {
            break;
        }

        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(graph.var(la::vector<double>(frames[i])));
        }

        lstm_var_tree = tensor_tree::make_var_tree(graph, param);
        pred_var_tree = tensor_tree::make_var_tree(graph, pred_param);

        if (ebt::in(std::string("dropout"), args)) {
            nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
                graph, lstm_var_tree, inputs, gen, dropout);
        } else {
            nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, inputs);
        }

        pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

        auto topo_order = autodiff::topo_order(pred_nn.logprob);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        double loss_sum = 0;
        double nframes = 0;

        for (int t = 0; t < pred_nn.logprob.size(); ++t) {
            auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob[t]);
            la::vector<double> gold;
            gold.resize(label_id.size());
            if (!ebt::in(labels[t], ignored)) {
                gold(label_id.at(labels[t])) = 1;
            }
            nn::log_loss loss { gold, pred };
            pred_nn.logprob[t]->grad = std::make_shared<la::vector<double>>(loss.grad());
            if (std::isnan(loss.loss())) {
                std::cerr << "loss is nan" << std::endl;
                exit(1);
            } else {
                loss_sum += loss.loss();
                nframes += 1;
            }
        }

        std::cout << "loss: " << loss_sum / nframes << std::endl;

        autodiff::grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = lstm::make_stacked_bi_lstm_tensor_tree(layer);
        tensor_tree::copy_grad(grad, lstm_var_tree);
        std::shared_ptr<tensor_tree::vertex> pred_grad = nn::make_pred_tensor_tree();
        tensor_tree::copy_grad(pred_grad, pred_var_tree);

        if (ebt::in(std::string("clip"), args)) {
            double n = tensor_tree::norm(grad);

            if (n > clip) {
                tensor_tree::imul(grad, clip / n);
                std::cout << "norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
            }
        }

        if (ebt::in(std::string("decay"), args)) {
            tensor_tree::rmsprop_update(param, grad, opt_data, decay, step_size);
            tensor_tree::rmsprop_update(pred_param, pred_grad, pred_opt_data, decay, step_size);
        } else {
            tensor_tree::adagrad_update(param, grad, opt_data, step_size);
            tensor_tree::adagrad_update(pred_param, pred_grad, pred_opt_data, step_size);
        }

        if (i % save_every == 0) {
            std::ofstream param_ofs { "param-last" };
            param_ofs << layer << std::endl;
            tensor_tree::save_tensor(param, param_ofs);
            tensor_tree::save_tensor(pred_param, param_ofs);
            param_ofs.close();

            std::ofstream opt_data_ofs { "opt-data-last" };
            opt_data_ofs << layer << std::endl;
            tensor_tree::save_tensor(opt_data, opt_data_ofs);
            tensor_tree::save_tensor(pred_opt_data, opt_data_ofs);
            opt_data_ofs.close();
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    tensor_tree::save_tensor(pred_param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt_data_ofs << layer << std::endl;
    tensor_tree::save_tensor(opt_data, opt_data_ofs);
    tensor_tree::save_tensor(pred_opt_data, opt_data_ofs);
    opt_data_ofs.close();
}

