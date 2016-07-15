#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/pred.h"
#include "nn/tensor_tree.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    std::shared_ptr<tensor_tree::vertex> param;
    lstm::stacked_bi_lstm_nn_t nn;

    std::shared_ptr<tensor_tree::vertex> pred_param;
    rnn::pred_nn_t pred_nn;

    int layer;
    std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
    std::shared_ptr<tensor_tree::vertex> pred_var_tree;

    std::vector<std::string> label;

    double dropout;

    int subsample_freq;
    int subsample_shift;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-lstm",
        "Predict frames with LSTM",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"dropout", "", false},
            {"light-dropout", "", false},
            {"logprob", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
            {"upsample", "", false},
            {"clockwork", "", false},
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

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);

    param = lstm::make_stacked_bi_lstm_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);

    pred_param = nn::make_pred_tensor_tree();
    tensor_tree::load_tensor(pred_param, param_ifs);
    param_ifs.close();

    label = speech::load_label_set(args.at("label"));

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

void prediction_env::run()
{
    int i = 1;

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

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

        lstm::lstm_builder *builder;

        if (ebt::in(std::string("clockwork"), args)) {
            la::vector<double> one_vec;
            one_vec.resize(tensor_tree::get_matrix(param->children[0]->children[0]->children[0]).rows(), 1);
            std::shared_ptr<autodiff::op_t> one = graph.var(one_vec);

            std::vector<std::shared_ptr<autodiff::op_t>> mask;

            for (int i = 0; i < inputs.size(); ++i) {
                la::vector<double> mask_vec;
                mask_vec.resize(one_vec.size());

                for (int d = 0; d < mask_vec.size(); ++d) {
                    if (d < mask_vec.size() / 2.0) {
                        mask_vec(d) = 0;
                    } else if (mask_vec.size() / 2.0 <= d && d < mask_vec.size() * 3 / 4.0) {
                        if (i % 2 == 0) {
                            mask_vec(d) = 0;
                        } else {
                            mask_vec(d) = 1;
                        }
                    } else {
                        if (i % 4 == 0) {
                            mask_vec(d) = 0;
                        } else {
                            mask_vec(d) = 1;
                        }
                    }
                }

                mask.push_back(graph.var(mask_vec));
            }

            builder = new lstm::zoneout_lstm_builder { mask, one };
        } else {
            builder = new lstm::lstm_builder{};
        }

        if (ebt::in(std::string("dropout"), args)) {
            if (ebt::in(std::string("light-dropout"), args)) {
                nn = lstm::make_stacked_bi_lstm_nn_with_dropout_light(
                    graph, lstm_var_tree, inputs, *builder, dropout);
            } else {
                nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
                    graph, lstm_var_tree, inputs, *builder, dropout);
            }
        } else {
            nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, inputs, *builder);
        }

        pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

        std::vector<std::shared_ptr<autodiff::op_t>> logprobs = pred_nn.logprob;

        if (ebt::in(std::string("upsample"), args)) {
            logprobs = lstm::upsample(pred_nn.logprob, subsample_freq, subsample_shift, frames.size());
        }

        auto topo_order = autodiff::topo_order(logprobs);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        std::cout << i << ".phn" << std::endl;

        if (ebt::in(std::string("logprob"), args)) {
            for (int t = 0; t < logprobs.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(logprobs[t]);

                std::cout << pred(0);

                for (int j = 1; j < pred.size(); ++j) {
                    std::cout << " " << pred(j);
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < logprobs.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(logprobs[t]);

                int argmax = -1;
                double max = -std::numeric_limits<double>::infinity();

                for (int j = 0; j < pred.size(); ++j) {
                    if (pred(j) > max) {
                        max = pred(j);
                        argmax = j;
                    }
                }

                std::cout << label[argmax] << std::endl;
            }
        }

        std::cout << "." << std::endl;

        delete builder;

        ++i;
    }
}

