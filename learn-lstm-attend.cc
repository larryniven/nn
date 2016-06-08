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
#include "nn/attention.h"
#include <random>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    lstm::dblstm_feat_param_t param;
    lstm::dblstm_feat_param_t opt_data;

    nn::pred_param_t pred_param;
    nn::pred_param_t pred_opt_data;

    lstm::dblstm_feat_nn_t nn;
    rnn::pred_nn_t pred_nn;

    double step_size;
    double rmsprop_decay;
    double momentum;

    double rnndrop_prob;

    int subsample_freq;
    int subsample_shift;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    int seed;

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
            {"rnndrop-prob", "", false},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"rnndrop-seed", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
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
    param = lstm::load_dblstm_feat_param(param_ifs);
    pred_param = nn::load_pred_param(param_ifs);
    param_ifs.close();

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt_data = lstm::load_dblstm_feat_param(opt_data_ifs);
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

    if (ebt::in(std::string("rnndrop-prob"), args)) {
        rnndrop_prob = std::stod(args.at("rnndrop-prob"));
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

    seed = 1;
    if (ebt::in(std::string("rnndrop-seed"), args)) {
        seed = std::stoi(args.at("rnndrop-seed"));
    }

    subsample_freq = 1;
    if (ebt::in(std::string("subsample-freq"), args)) {
        subsample_freq = std::stoi(args.at("subsample-freq"));
    }

    subsample_shift = 0;
    if (ebt::in(std::string("subsample-shift"), args)) {
        subsample_shift = std::stoi(args.at("subsample-shift"));
    }

    if (ebt::in(std::string("ignored"), args)) {
        auto parts = ebt::split(args.at("ignored"), ",");
        ignored.insert(parts.begin(), parts.end());
    }
}

void learning_env::run()
{
    int i = 1;

    std::default_random_engine gen { seed };

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

        std::vector<std::shared_ptr<autodiff::op_t>> subsampled_inputs
            = rnn::subsample_input(inputs, subsample_freq, subsample_shift);

        nn = lstm::make_dblstm_feat_nn(graph, param, subsampled_inputs);

        if (ebt::in(std::string("rnndrop-prob"), args)) {
            lstm::apply_random_mask(nn, param, gen, rnndrop_prob);
        }

        std::shared_ptr<autodiff::op_t> hs = autodiff::row_cat(nn.layer.back().output);
        std::vector<attention::attention_nn_t> atts;
        std::vector<std::shared_ptr<autodiff::op_t>> context;
        
        for (int i = 0; i < nn.layer.back().output.size(); ++i) {
            atts.push_back(attention::attend(hs, nn.layer.back().output[i]));
            context.push_back(atts.back().context);
        }

        pred_nn = rnn::make_pred_nn(graph, pred_param, context);

        std::vector<std::shared_ptr<autodiff::op_t>> upsampled_output
            = rnn::upsample_output(pred_nn.logprob, subsample_freq, subsample_shift, frames.size());

        assert(upsampled_output.size() == frames.size());

        auto topo_order = autodiff::topo_order(upsampled_output);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        double loss_sum = 0;
        double nframes = 0;

        for (int t = 0; t < upsampled_output.size(); ++t) {
            auto& pred = autodiff::get_output<la::vector<double>>(upsampled_output[t]);
            la::vector<double> gold;
            gold.resize(label_id.size());
            if (!ebt::in(labels[t], ignored)) {
                gold(label_id.at(labels[t])) = 1;
            }
            nn::log_loss loss { gold, pred };
            upsampled_output[t]->grad = std::make_shared<la::vector<double>>(loss.grad());
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

        lstm::dblstm_feat_param_t grad = lstm::copy_dblstm_feat_grad(nn);
        nn::pred_param_t pred_grad = rnn::copy_grad(pred_nn);

        if (ebt::in(std::string("momentum"), args)) {
            // lstm::const_step_update_momentum(param, grad, opt_data, momentum, step_size);
            std::cerr << "not implemented" << std::endl;
            exit(1);
        } else if (ebt::in(std::string("rmsprop-decay"), args)) {
            lstm::rmsprop_update(param, grad, opt_data, rmsprop_decay, step_size);
            nn::rmsprop_update(pred_param, pred_grad, pred_opt_data, rmsprop_decay, step_size);
        } else {
            lstm::adagrad_update(param, grad, opt_data, step_size);
            nn::adagrad_update(pred_param, pred_grad, pred_opt_data, step_size);
        }

#if 0
        {
            lstm::param_t p = param;
            p.hidden_input(0, 0) += 1e-8;
            lstm::nn_t nn2 = lstm::make_nn(p, frames);
            lstm::eval(nn2);
            auto& pred = autodiff::get_output<la::vector<double>>(nn2.logprob.at(1));
            la::vector<double> gold;
            gold.resize(label_id.size());
            gold(label_id.at(labels[1])) = 1;
            lstm::log_loss loss2 { gold, pred };

            auto& grad = autodiff::get_grad<la::matrix<double>>(nn.hidden_input);
            std::cout << (loss2.loss() - loss_1) / 1e-8 << " " << grad(0, 0) << std::endl;
        }
#endif

        if (i % save_every == 0) {
            std::ofstream param_ofs { "param-last" };
            lstm::save_dblstm_feat_param(param, param_ofs);
            nn::save_pred_param(pred_param, param_ofs);
            param_ofs.close();

            std::ofstream opt_data_ofs { "opt-data-last" };
            lstm::save_dblstm_feat_param(opt_data, opt_data_ofs);
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
    lstm::save_dblstm_feat_param(param, param_ofs);
    nn::save_pred_param(pred_param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    lstm::save_dblstm_feat_param(opt_data, opt_data_ofs);
    nn::save_pred_param(pred_opt_data, opt_data_ofs);
    opt_data_ofs.close();
}

