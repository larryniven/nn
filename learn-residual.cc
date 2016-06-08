#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "opt/opt.h"
#include "nn/residual.h"
#include "nn/pred.h"
#include "nn/nn.h"
#include <random>
#include <algorithm>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    residual::nn_param_t param;
    residual::nn_param_t opt_data;

    nn::pred_param_t pred_param;
    nn::pred_param_t pred_opt_data;

    residual::nn_t nn;
    nn::pred_nn_t pred_nn;

    double step_size;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

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
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignored", "", false}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::ifstream param_ifs { args.at("param") };
    param = residual::load_nn_param(param_ifs);
    pred_param = nn::load_pred_param(param_ifs);
    param_ifs.close();

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt_data = residual::load_nn_param(opt_data_ifs);
    pred_opt_data = nn::load_pred_param(opt_data_ifs);
    opt_data_ifs.close();

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    step_size = std::stod(args.at("step-size"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    std::vector<std::string> label_vec = speech::load_label_set(args.at("label"));
    for (int i = 0; i < label_vec.size(); ++i) {
        label_id[label_vec[i]] = i;
    }

    if (ebt::in(std::string("ignored"), args)) {
        auto parts = ebt::split(args.at("ignored"), ",");
        ignored.insert(parts.begin(), parts.end());
    }
}

void learning_env::run()
{
    int i = 1;

    std::default_random_engine gen;

    residual::nn_param_t zero_grad;
    residual::resize_as(zero_grad, param);

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

	std::vector<std::string> labels;

	labels = speech::load_label_batch(label_batch);

        if (!frame_batch || !label_batch) {
            break;
        }

        std::vector<int> indices;
        indices.resize(frames.size());
        for (int j = 0; j < frames.size(); ++j) {
            indices[j] = j;
        }
        std::shuffle(indices.begin(), indices.end(), gen);

        double loss_sum = 0;
        double nframes = 0;

        for (int j = 0; j < indices.size(); ++j) {
            int t = indices[j];

            if (ebt::in(labels[t], ignored)) {
                continue;
            }

            autodiff::computation_graph graph;

            nn = residual::make_nn(graph, param);
            pred_nn = nn::make_pred_nn(graph, nn.layer.back().output, pred_param);

            std::vector<double> input;

            for (int k = -5; k <= 5; ++k) {
                if (0 <= k + t && k + t < frames.size()) {
                    input.insert(input.end(), frames[k + t].begin(), frames[k + t].end());
                } else {
                    input.resize(input.size() + frames.front().size());
                }
            }

            nn.input->output = std::make_shared<la::vector<double>>(la::vector<double> { std::move(input) });
            autodiff::eval(pred_nn.logprob, autodiff::eval_funcs);

            auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob);
            la::vector<double> gold;
            gold.resize(label_id.size());
            if (!ebt::in(labels[t], ignored)) {
                gold(label_id.at(labels[t])) = 1;
            }
            nn::log_loss loss { gold, pred };
            pred_nn.logprob->grad = std::make_shared<la::vector<double>>(loss.grad());

            if (std::isnan(loss.loss())) {
                std::cerr << "loss is nan" << std::endl;
                exit(1);
            } else {
                loss_sum += loss.loss();
                nframes += 1;
            }

            residual::nn_param_t grad = zero_grad;
            residual::nn_tie_grad(nn, grad);

            autodiff::grad(pred_nn.logprob, autodiff::grad_funcs);

            nn::pred_param_t pred_grad = nn::copy_grad(pred_nn);

            residual::adagrad_update(param, grad, opt_data, step_size);
            nn::adagrad_update(pred_param, pred_grad, pred_opt_data, step_size);
        }

        std::cout << "loss: " << loss_sum / nframes << std::endl;

        if (i % save_every == 0) {
            std::ofstream param_ofs { "param-last" };
            residual::save_nn_param(param, param_ofs);
            nn::save_pred_param(pred_param, param_ofs);
            param_ofs.close();

            std::ofstream opt_data_ofs { "opt-data-last" };
            residual::save_nn_param(opt_data, opt_data_ofs);
            nn::save_pred_param(pred_opt_data, opt_data_ofs);
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
    residual::save_nn_param(param, param_ofs);
    nn::save_pred_param(pred_param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    residual::save_nn_param(opt_data, opt_data_ofs);
    nn::save_pred_param(pred_opt_data, opt_data_ofs);
    opt_data_ofs.close();
}

