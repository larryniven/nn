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
    std::shared_ptr<tensor_tree::vertex> param_first_moment;
    std::shared_ptr<tensor_tree::vertex> param_second_moment;

    std::shared_ptr<tensor_tree::vertex> pred_param;
    std::shared_ptr<tensor_tree::vertex> pred_opt_data;
    std::shared_ptr<tensor_tree::vertex> pred_first_moment;
    std::shared_ptr<tensor_tree::vertex> pred_second_moment;

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

    int subsample_freq;
    int subsample_shift;

    int time;
    double adam_beta1;
    double adam_beta2;

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
            {"dropout-seed", "", false},
            {"light-dropout", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
            {"adam-beta1", "", false},
            {"adam-beta2", "", false},
            {"clockwork", "", false}
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

    if (ebt::in(std::string("adam-beta1"), args)) {
        time = std::stoi(line);
        std::getline(opt_data_ifs, line);
        param_first_moment = lstm::make_stacked_bi_lstm_tensor_tree(layer);
        tensor_tree::load_tensor(param_first_moment, opt_data_ifs);
        param_second_moment = lstm::make_stacked_bi_lstm_tensor_tree(layer);
        tensor_tree::load_tensor(param_second_moment, opt_data_ifs);
        pred_first_moment = nn::make_pred_tensor_tree();
        tensor_tree::load_tensor(pred_first_moment, opt_data_ifs);
        pred_second_moment = nn::make_pred_tensor_tree();
        tensor_tree::load_tensor(pred_second_moment, opt_data_ifs);
    } else {
        opt_data = lstm::make_stacked_bi_lstm_tensor_tree(layer);
        tensor_tree::load_tensor(opt_data, opt_data_ifs);
        pred_opt_data = nn::make_pred_tensor_tree();
        tensor_tree::load_tensor(pred_opt_data, opt_data_ifs);
    }
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

    subsample_freq = 1;
    if (ebt::in(std::string("subsample-freq"), args)) {
        subsample_freq = std::stoi(args.at("subsample-freq"));
    }

    if (ebt::in(std::string("subsample-shift"), args)) {
        subsample_shift = std::stoi(args.at("subsample-shift"));
    }

    if (ebt::in(std::string("adam-beta1"), args)) {
        adam_beta1 = std::stod(args.at("adam-beta1"));
    }

    if (ebt::in(std::string("adam-beta2"), args)) {
        adam_beta2 = std::stod(args.at("adam-beta2"));
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
                    graph, lstm_var_tree, inputs, *builder, gen, dropout);
            } else {
                nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
                    graph, lstm_var_tree, inputs, *builder, gen, dropout);
            }
        } else {
            nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, inputs, *builder);
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

            autodiff::grad(topo_order, autodiff::grad_funcs);
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

            autodiff::grad(topo_order, autodiff::grad_funcs);
        }

        std::cout << "loss: " << loss_sum / nframes << std::endl;

        std::shared_ptr<tensor_tree::vertex> grad = lstm::make_stacked_bi_lstm_tensor_tree(layer);
        tensor_tree::copy_grad(grad, lstm_var_tree);
        std::shared_ptr<tensor_tree::vertex> pred_grad = nn::make_pred_tensor_tree();
        tensor_tree::copy_grad(pred_grad, pred_var_tree);

        if (ebt::in(std::string("clip"), args)) {
            double n1 = tensor_tree::norm(grad);
            double n2 = tensor_tree::norm(pred_grad);
            double n = std::sqrt(n1 * n1 + n2 * n2);

            if (n > clip) {
                tensor_tree::imul(grad, clip / n);
                tensor_tree::imul(pred_grad, clip / n);
                std::cout << "norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
            }
        }

        double v1 = tensor_tree::get_matrix(param->children[0]->children[0]->children[0])(0, 0);

        if (ebt::in(std::string("decay"), args)) {
            tensor_tree::rmsprop_update(param, grad, opt_data, decay, step_size);
            tensor_tree::rmsprop_update(pred_param, pred_grad, pred_opt_data, decay, step_size);
        } if (ebt::in(std::string("adam-beta1"), args)) {
            tensor_tree::adam_update(param, grad, param_first_moment, param_second_moment,
                time, step_size, adam_beta1, adam_beta2);
            tensor_tree::adam_update(pred_param, pred_grad, pred_first_moment, pred_second_moment,
                time, step_size, adam_beta1, adam_beta2);
        } else {
            tensor_tree::adagrad_update(param, grad, opt_data, step_size);
            tensor_tree::adagrad_update(pred_param, pred_grad, pred_opt_data, step_size);
        }

        double v2 = tensor_tree::get_matrix(param->children[0]->children[0]->children[0])(0, 0);

        std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        if (i % save_every == 0) {
            std::ofstream param_ofs { "param-last" };
            param_ofs << layer << std::endl;
            tensor_tree::save_tensor(param, param_ofs);
            tensor_tree::save_tensor(pred_param, param_ofs);
            param_ofs.close();

            if (ebt::in(std::string("adam-beta1"), args)) {
                std::ofstream opt_data_ofs { "opt-data-last" };
                opt_data_ofs << time << std::endl;
                opt_data_ofs << layer << std::endl;
                tensor_tree::save_tensor(param_first_moment, opt_data_ofs);
                tensor_tree::save_tensor(param_second_moment, opt_data_ofs);
                tensor_tree::save_tensor(pred_first_moment, opt_data_ofs);
                tensor_tree::save_tensor(pred_second_moment, opt_data_ofs);
                opt_data_ofs.close();
            } else {
                std::ofstream opt_data_ofs { "opt-data-last" };
                opt_data_ofs << layer << std::endl;
                tensor_tree::save_tensor(opt_data, opt_data_ofs);
                tensor_tree::save_tensor(pred_opt_data, opt_data_ofs);
                opt_data_ofs.close();
            }
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        delete builder;

        ++i;
        ++time;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    tensor_tree::save_tensor(pred_param, param_ofs);
    param_ofs.close();

    if (ebt::in(std::string("adam-beta1"), args)) {
        std::ofstream opt_data_ofs { output_opt_data };
        opt_data_ofs << time << std::endl;
        opt_data_ofs << layer << std::endl;
        tensor_tree::save_tensor(param_first_moment, opt_data_ofs);
        tensor_tree::save_tensor(param_second_moment, opt_data_ofs);
        tensor_tree::save_tensor(pred_first_moment, opt_data_ofs);
        tensor_tree::save_tensor(pred_second_moment, opt_data_ofs);
        opt_data_ofs.close();
    } else {
        std::ofstream opt_data_ofs { output_opt_data };
        opt_data_ofs << layer << std::endl;
        tensor_tree::save_tensor(opt_data, opt_data_ofs);
        tensor_tree::save_tensor(pred_opt_data, opt_data_ofs);
        opt_data_ofs.close();
    }
}

