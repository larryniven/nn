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

struct loss_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::vertex> pred_param;

    std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
    std::shared_ptr<tensor_tree::vertex> pred_var_tree;

    int layer;

    lstm::stacked_bi_lstm_nn_t nn;
    rnn::pred_nn_t pred_nn;

    double dropout;

    int subsample_freq;
    int subsample_shift;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    loss_env(std::unordered_map<std::string, std::string> args);

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
            {"label", "", true},
            {"ignored", "", false},
            {"dropout", "", false},
            {"light-dropout", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
            {"upsample", "", false},
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

    loss_env env { args };

    env.run();

    return 0;
}

loss_env::loss_env(std::unordered_map<std::string, std::string> args)
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

    subsample_freq = 1;
    if (ebt::in(std::string("subsample-freq"), args)) {
        subsample_freq = std::stoi(args.at("subsample-freq"));
    }

    if (ebt::in(std::string("subsample-shift"), args)) {
        subsample_shift = std::stoi(args.at("subsample-shift"));
    }
}

void loss_env::run()
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

        assert(frames.size() == labels.size());

        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(graph.var(la::vector<double>(frames[i])));
        }

        if (subsample_freq > 1) {
            inputs = lstm::subsample(inputs, subsample_freq, subsample_shift);
        }

        lstm_var_tree = tensor_tree::make_var_tree(graph, param);
        pred_var_tree = tensor_tree::make_var_tree(graph, pred_param);

        if (ebt::in(std::string("dropout"), args)) {
            if (ebt::in(std::string("light-dropout"), args)) {
                nn = lstm::make_stacked_bi_lstm_nn_with_dropout_light(
                    graph, lstm_var_tree, inputs, lstm::lstm_builder{}, dropout);
            } else {
                nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
                    graph, lstm_var_tree, inputs, lstm::lstm_builder{}, dropout);
            }
        } else {
            nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, inputs, lstm::lstm_builder{});
        }

        pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

        double loss_sum = 0;
        double nframes = 0;

        if (ebt::in(std::string("upsample"), args)) {
            std::vector<std::shared_ptr<autodiff::op_t>> logprobs
                = lstm::upsample(pred_nn.logprob, subsample_freq, subsample_shift, labels.size());

            auto topo_order = autodiff::topo_order(logprobs);
            autodiff::eval(topo_order, autodiff::eval_funcs);

            assert(labels.size() == logprobs.size());

            for (int t = 0; t < labels.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(logprobs[t]);
                la::vector<double> gold;
                gold.resize(label_id.size());
                if (!ebt::in(labels[t], ignored)) {
                    gold(label_id.at(labels[t])) = 1;
                }
                nn::log_loss loss { gold, pred };
                logprobs[t]->grad = std::make_shared<la::vector<double>>(loss.grad());
                if (std::isnan(loss.loss())) {
                    std::cerr << "loss is nan" << std::endl;
                    exit(1);
                } else {
                    loss_sum += loss.loss();
                    nframes += 1;
                }
            }
        } else {
            std::vector<std::shared_ptr<autodiff::op_t>> logprobs = pred_nn.logprob;

            auto topo_order = autodiff::topo_order(logprobs);
            autodiff::eval(topo_order, autodiff::eval_funcs);

            std::vector<std::string> subsampled_labels
                = lstm::subsample(labels, subsample_freq, subsample_shift);

            assert(subsampled_labels.size() == logprobs.size());

            for (int t = 0; t < subsampled_labels.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(logprobs[t]);
                la::vector<double> gold;
                gold.resize(label_id.size());
                if (!ebt::in(subsampled_labels[t], ignored)) {
                    gold(label_id.at(subsampled_labels[t])) = 1;
                }
                nn::log_loss loss { gold, pred };
                logprobs[t]->grad = std::make_shared<la::vector<double>>(loss.grad());
                if (std::isnan(loss.loss())) {
                    std::cerr << "loss is nan" << std::endl;
                    exit(1);
                } else {
                    loss_sum += loss.loss();
                    nframes += 1;
                }
            }
        }

        std::cout << "loss: " << loss_sum / nframes << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }
}

