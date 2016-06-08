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

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    lstm::db_lstm2d_param_t param;
    lstm::db_lstm2d_param_t opt_data;

    nn::pred_param_t pred_param;
    nn::pred_param_t pred_opt_data;

    lstm::db_lstm2d_nn_t nn;
    rnn::pred_nn_t pred_nn;

    double step_size;
    double rmsprop_decay;
    double momentum;

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
            {"rmsprop-decay", "", false},
            {"momentum", "", false},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignored", "", false},
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
    param = lstm::load_db_lstm2d_param(param_ifs);
    pred_param = nn::load_pred_param(param_ifs);
    param_ifs.close();

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt_data = lstm::load_db_lstm2d_param(opt_data_ifs);
    pred_opt_data = nn::load_pred_param(opt_data_ifs);
    opt_data_ifs.close();

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("momentum"), args)) {
        momentum = std::stod(args.at("momentum"));
    }

    if (ebt::in(std::string("rmsprop-decay"), args)) {
        rmsprop_decay = std::stod(args.at("rmsprop-decay"));
    }

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

        nn = make_db_lstm2d_nn(graph, param, inputs);

        pred_nn = rnn::make_pred_nn(graph, pred_param, nn.layer.back().output);

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

#if 0
        {
            auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob[0]);
            la::vector<double> gold;
            gold.resize(label_id.size());
            gold(label_id.at(labels[0])) = 1;
            nn::log_loss loss { gold, pred };

            double ell1 = loss.loss();

            autodiff::computation_graph graph2;
            lstm::bi_lstm2d_param_t param2 = param;
            nn::pred_param_t pred_param2 = pred_param;
            pred_param2.softmax_bias(0) += 1e-8;

            std::vector<std::shared_ptr<autodiff::op_t>> inputs2;

            for (int i = 0; i < frames.size(); ++i) {
                inputs2.push_back(graph2.var(la::vector<double>(frames[i])));
            }

            lstm::bi_lstm2d_nn_t nn2 = make_bi_lstm2d_nn(graph2, param2, inputs2, layer);

            rnn::pred_nn_t pred_nn2 = rnn::make_pred_nn(graph2, pred_param2, nn2.output);

            auto topo_order2 = autodiff::topo_order(pred_nn2.logprob[0]);
            autodiff::eval(topo_order2, autodiff::eval_funcs);

            auto& pred2 = autodiff::get_output<la::vector<double>>(pred_nn2.logprob[0]);
            la::vector<double> gold2;
            gold2.resize(label_id.size());
            gold2(label_id.at(labels[0])) = 1;
            nn::log_loss loss2 { gold2, pred2 };

            double ell2 = loss2.loss();

            pred_nn2.logprob[0]->grad = std::make_shared<la::vector<double>>(loss2.grad());
            autodiff::grad(topo_order2, autodiff::grad_funcs);

            auto& grad = autodiff::get_grad<la::vector<double>>(pred_nn2.softmax_bias);
            std::cout << "analytical grad: " << grad(0) << std::endl;

            std::cout << "numeric grad: " << (ell2 - ell1) / 1e-8 << std::endl;
        }
#endif

        std::cout << "loss: " << loss_sum / nframes << std::endl;

        autodiff::grad(topo_order, autodiff::grad_funcs);

        lstm::db_lstm2d_param_t grad = lstm::copy_db_lstm2d_grad(nn);
        nn::pred_param_t pred_grad = rnn::copy_grad(pred_nn);

        if (ebt::in(std::string("rmsprop-decay"), args)) {
            lstm::rmsprop_update(param, grad, opt_data, rmsprop_decay, step_size);
            nn::rmsprop_update(pred_param, pred_grad, pred_opt_data, rmsprop_decay, step_size);
        } else {
            lstm::adagrad_update(param, grad, opt_data, step_size);
            nn::adagrad_update(pred_param, pred_grad, pred_opt_data, step_size);
        }

        if (i % save_every == 0) {
            std::ofstream param_ofs { "param-last" };
            lstm::save_db_lstm2d_param(param, param_ofs);
            nn::save_pred_param(pred_param, param_ofs);
            param_ofs.close();

            std::ofstream opt_data_ofs { "opt-data-last" };
            lstm::save_db_lstm2d_param(opt_data, opt_data_ofs);
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
    lstm::save_db_lstm2d_param(param, param_ofs);
    nn::save_pred_param(pred_param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    lstm::save_db_lstm2d_param(opt_data, opt_data_ofs);
    nn::save_pred_param(pred_opt_data, opt_data_ofs);
    opt_data_ofs.close();
}

