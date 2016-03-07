#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "opt/opt.h"
#include "nn/gru.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    gru::dbgru_feat_param_t dbgru_param;
    gru::dbgru_feat_param_t dbgru_opt_data;
    gru::pred_param_t pred_param;
    gru::pred_param_t pred_opt_data;

    double step_size;
    double rmsprop_decay;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-gru",
        "Train a GRU frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"rmsprop-decay", "", false},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
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
    dbgru_param = gru::load_dbgru_feat_param(param_ifs);
    pred_param = gru::load_pred_param(param_ifs);

    std::ifstream opt_data_ifs { args.at("opt-data") };
    dbgru_opt_data = gru::load_dbgru_feat_param(opt_data_ifs);
    pred_opt_data = gru::load_pred_param(opt_data_ifs);

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    step_size = std::stod(args.at("step-size"));

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

        autodiff::computation_graph g;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;
        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(g.var(la::vector<double>(frames[i])));
        }

        gru::dbgru_feat_nn_t dbgru_feat_nn = gru::make_dbgru_feat_nn(g, dbgru_param, inputs);
        gru::pred_nn_t pred_nn = gru::make_pred_nn(g, pred_param, dbgru_feat_nn.layer.back().output);

        gru::eval(pred_nn);

        double loss_sum = 0;
        double nframes = 0;

        for (int t = 0; t < pred_nn.logprob.size(); ++t) {
            auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob.at(t));
            la::vector<double> gold;
            gold.resize(label_id.size());
            gold(label_id.at(labels[t])) = 1;
            gru::log_loss loss { gold, pred };
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

        gru::grad(pred_nn);

        gru::dbgru_feat_param_t dbgru_grad = gru::copy_grad(dbgru_feat_nn);
        gru::pred_param_t pred_grad = gru::copy_grad(pred_nn);

        if (ebt::in(std::string("rmsprop-decay"), args)) {
            gru::rmsprop_update(dbgru_param, dbgru_grad, dbgru_opt_data, rmsprop_decay, step_size);
            gru::rmsprop_update(pred_param, pred_grad, pred_opt_data, rmsprop_decay, step_size);
        } else {
            gru::adagrad_update(dbgru_param, dbgru_grad, dbgru_opt_data, step_size);
            gru::adagrad_update(pred_param, pred_grad, pred_opt_data, step_size);
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    std::ofstream output_param_ofs { output_param };
    gru::save_dbgru_feat_param(dbgru_param, output_param_ofs);
    gru::save_pred_param(pred_param, output_param_ofs);
    output_param_ofs.close();

    std::ofstream output_opt_data_ofs { output_opt_data };
    gru::save_dbgru_feat_param(dbgru_opt_data, output_opt_data_ofs);
    gru::save_pred_param(pred_opt_data, output_opt_data_ofs);
    output_opt_data_ofs.close();
}

