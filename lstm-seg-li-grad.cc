#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
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

    std::shared_ptr<tensor_tree::vertex> grad_avg;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    int layer;

    lstm::stacked_bi_lstm_nn_t nn;

    double dropout;
    int dropout_seed;

    std::default_random_engine gen;

    std::string output;

    double clip;

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
        "grad-lstm-seg-li",
        "Compute gradient of a LSTM frame classifier",
        {
            {"frame-batch", "", true},
            {"gt-batch", "", true},
            {"param", "", true},
            {"output", "", false},
            {"clip", "", false},
            {"label", "", true},
            {"ignored", "", false},
            {"dropout", "", false},
            {"dropout-seed", "", false},
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

    grad_avg = lstm_seg::make_tensor_tree(layer, args);
    tensor_tree::resize_as(grad_avg, param);
}

void learning_env::run()
{
    int sample = 0;

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

	std::vector<speech::segment> segs = speech::load_segments(gt_batch);

        if (!frame_batch || !gt_batch) {
            break;
        }

        std::cout << "utterance: " << sample + 1 << std::endl;

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

        ++sample;
    }

    tensor_tree::imul(grad_avg, 1.0 / sample);

    std::ofstream ofs { args.at("output") };
    ofs << layer << std::endl;
    tensor_tree::save_tensor(grad_avg, ofs);
    ofs.close();
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

    tensor_tree::iadd(grad_avg, grad);
}

