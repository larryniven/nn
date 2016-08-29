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
    std::shared_ptr<tensor_tree::vertex> opt_data;
    std::shared_ptr<tensor_tree::vertex> param_first_moment;
    std::shared_ptr<tensor_tree::vertex> param_second_moment;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    int layer;

    lstm::stacked_bi_lstm_nn_t nn;

    double step_size;
    double decay;

    double dropout;
    int dropout_seed;

    std::default_random_engine gen;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    double clip;

    int time;
    double adam_beta1;
    double adam_beta2;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

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
        "lstm-seg-li-learn",
        "Train a LSTM frame classifier",
        {
            {"frame-batch", "", true},
            {"gt-batch", "", true},
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
            {"adam-beta1", "", false},
            {"adam-beta2", "", false},
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
    frame_batch.open(args.at("frame-batch"));
    gt_batch.open(args.at("gt-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_seg::make_tensor_tree(layer, args);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);

    if (ebt::in(std::string("adam-beta1"), args)) {
        time = std::stoi(line);
        std::getline(opt_data_ifs, line);
        param_first_moment = lstm_seg::make_tensor_tree(layer, args);
        tensor_tree::load_tensor(param_first_moment, opt_data_ifs);
        param_second_moment = lstm_seg::make_tensor_tree(layer, args);
        tensor_tree::load_tensor(param_second_moment, opt_data_ifs);
    } else {
        opt_data = lstm_seg::make_tensor_tree(layer, args);
        tensor_tree::load_tensor(opt_data, opt_data_ifs);
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
        gen.seed(dropout_seed);
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

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

	std::vector<speech::segment> segs = speech::load_segments(gt_batch);

        if (!frame_batch || !gt_batch) {
            break;
        }

        std::cout << "utterance: " << i << std::endl;

        for (int i = 0; i < segs.size(); ++i) {
            if (ebt::in(segs[i].label, ignored)) {
                continue;
            }

            std::vector<std::vector<double>> seg_frames;

            for (int t = std::max<int>(0, segs[i].start_time - 3);
                    t < std::min<int>(segs[i].end_time + 3, frames.size()); ++t) {
                seg_frames.push_back(frames[t]);
            }

            if (seg_frames.size() == 0) {
                continue;
            }

            learn_sample(seg_frames, segs[i].label);
        }

        if (i % save_every == 0) {
            std::ofstream param_ofs { "param-last" };
            param_ofs << layer << std::endl;
            tensor_tree::save_tensor(param, param_ofs);
            param_ofs.close();

            if (ebt::in(std::string("adam-beta1"), args)) {
                std::ofstream opt_data_ofs { "opt-data-last" };
                opt_data_ofs << time << std::endl;
                opt_data_ofs << layer << std::endl;
                tensor_tree::save_tensor(param_first_moment, opt_data_ofs);
                tensor_tree::save_tensor(param_second_moment, opt_data_ofs);
                opt_data_ofs.close();
            } else {
                std::ofstream opt_data_ofs { "opt-data-last" };
                opt_data_ofs << layer << std::endl;
                tensor_tree::save_tensor(opt_data, opt_data_ofs);
                opt_data_ofs.close();
            }
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
        ++time;
    }

    std::ofstream param_ofs { output_param };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    if (ebt::in(std::string("adam-beta1"), args)) {
        std::ofstream opt_data_ofs { output_opt_data };
        opt_data_ofs << time << std::endl;
        opt_data_ofs << layer << std::endl;
        tensor_tree::save_tensor(param_first_moment, opt_data_ofs);
        tensor_tree::save_tensor(param_second_moment, opt_data_ofs);
        opt_data_ofs.close();
    } else {
        std::ofstream opt_data_ofs { output_opt_data };
        opt_data_ofs << layer << std::endl;
        tensor_tree::save_tensor(opt_data, opt_data_ofs);
        opt_data_ofs.close();
    }
}

void learning_env::learn_sample(
    std::vector<std::vector<double>> const& frames,
    std::string const& label)
{
    autodiff::computation_graph graph;
    std::vector<std::shared_ptr<autodiff::op_t>> inputs;

    for (int i = 0; i < frames.size(); ++i) {
        inputs.push_back(graph.var(la::vector<double>(frames[i])));
    }

    var_tree = tensor_tree::make_var_tree(graph, param);

    if (ebt::in(std::string("dropout"), args)) {
        nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
            graph, var_tree->children.front(), inputs, lstm::lstm_builder{}, gen, dropout);
    } else {
        nn = lstm::make_stacked_bi_lstm_nn(var_tree->children.front(), inputs, lstm::lstm_builder{});
    }

    std::shared_ptr<autodiff::op_t> pred_var;

    pred_var = lstm_seg::make_pred_nn(graph, nn, var_tree, param, args);

    auto topo_order = autodiff::topo_order(pred_var);
    autodiff::eval(topo_order, autodiff::eval_funcs);

    la::vector<double> gold;
    gold.resize(label_id.size());
    gold(label_id.at(label)) = 1;

    la::vector<double>& pred = autodiff::get_output<la::vector<double>>(pred_var);

    nn::log_loss loss { gold, pred };
    pred_var->grad = std::make_shared<la::vector<double>>(loss.grad());

    if (std::isnan(loss.loss())) {
        std::cerr << "loss is nan" << std::endl;
        exit(1);
    }

    std::cout << "loss: " << loss.loss() << std::endl;

    autodiff::grad(topo_order, autodiff::grad_funcs);

    std::shared_ptr<tensor_tree::vertex> grad = lstm_seg::make_tensor_tree(layer, args);
    tensor_tree::resize_as(grad, param);
    tensor_tree::copy_grad(grad, var_tree);

    if (ebt::in(std::string("clip"), args)) {
        double n = tensor_tree::norm(grad);

        if (n > clip) {
            tensor_tree::imul(grad, clip / n);
            std::cout << "norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
        }
    }

    double v1 = tensor_tree::get_matrix(param->children[0]->children[0]->children[0]->children[0])(0, 0);

    if (ebt::in(std::string("decay"), args)) {
        tensor_tree::rmsprop_update(param, grad, opt_data, decay, step_size);
    } if (ebt::in(std::string("adam-beta1"), args)) {
        tensor_tree::adam_update(param, grad, param_first_moment, param_second_moment,
            time, step_size, adam_beta1, adam_beta2);
    } else {
        tensor_tree::adagrad_update(param, grad, opt_data, step_size);
    }

    double v2 = tensor_tree::get_matrix(param->children[0]->children[0]->children[0]->children[0])(0, 0);

    std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

}

