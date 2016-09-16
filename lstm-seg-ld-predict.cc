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

struct prediction_env {

    std::ifstream frame_batch;
    std::ifstream gt_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    int layer;

    lstm::stacked_bi_lstm_nn_t nn;

    double dropout;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

    void predict_sample(
        std::vector<std::vector<double>> const& frames);

};

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer);

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-lstm-seg",
        "Predict segments with an LSTM ",
        {
            {"frame-batch", "", true},
            {"gt-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"dropout", "", false},
            {"print-attention", "", false},
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

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex v { tensor_tree::tensor_t::nil };
    v.children.push_back(lstm::make_stacked_bi_lstm_tensor_tree(layer));
    v.children.push_back(tensor_tree::make_matrix());
    v.children.push_back(tensor_tree::make_vector());
    return std::make_shared<tensor_tree::vertex>(v);
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));
    gt_batch.open(args.at("gt-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }
}

void prediction_env::run()
{
    int i = 1;

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

	std::vector<speech::segment> segs = speech::load_segments(gt_batch);

        if (!frame_batch || !gt_batch) {
            break;
        }

        std::cout << i << std::endl;

        for (int i = 0; i < segs.size(); ++i) {
            std::vector<std::vector<double>> seg_frames;

            for (int t = std::max<int>(0, segs[i].start_time - 3);
                    t < std::min<int>(segs[i].end_time + 3, frames.size()); ++t) {
                seg_frames.push_back(frames[t]);
            }

            if (seg_frames.size() == 0) {
                std::cout << "unk" << std::endl;
                continue;
            }

            predict_sample(seg_frames);
        }

        std::cout << "." << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }
}

void prediction_env::predict_sample(
    std::vector<std::vector<double>> const& frames)
{
    autodiff::computation_graph graph;
    std::vector<std::shared_ptr<autodiff::op_t>> inputs;

    for (int i = 0; i < frames.size(); ++i) {
        inputs.push_back(graph.var(la::vector<double>(frames[i])));
    }

    var_tree = tensor_tree::make_var_tree(graph, param);

    if (ebt::in(std::string("dropout"), args)) {
        nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
            graph, var_tree->children.front(), inputs, lstm::lstm_builder{}, dropout);
    } else {
        nn = lstm::make_stacked_bi_lstm_nn(var_tree->children.front(), inputs, lstm::lstm_builder{});
    }

    std::shared_ptr<autodiff::op_t> hs = autodiff::col_cat(nn.layer.back().output);
    std::shared_ptr<autodiff::op_t> pre_att = autodiff::mmul(tensor_tree::get_var(var_tree->children[1]), hs);
    std::vector<std::shared_ptr<autodiff::op_t>> feats;
    std::vector<std::shared_ptr<autodiff::op_t>> atts;

    for (int i = 0; i < label_id.size(); ++i) {
        std::shared_ptr<autodiff::op_t> att = autodiff::softmax(autodiff::row_at(pre_att, i));
        atts.push_back(att);
        feats.push_back(autodiff::mul(hs, att));
    }

    std::shared_ptr<autodiff::op_t> pred_var = autodiff::logsoftmax(
        autodiff::mul(autodiff::row_cat(feats), tensor_tree::get_var(var_tree->children[2])));

    auto topo_order = autodiff::topo_order(pred_var);
    autodiff::eval(topo_order, autodiff::eval_funcs);

    double max = -std::numeric_limits<double>::infinity();
    int argmax = -1;

    double min = std::numeric_limits<double>::infinity();
    int argmin = -1;

    auto& pred = autodiff::get_output<la::vector_like<double>>(pred_var);

    for (int i = 0; i < pred.size(); ++i) {
        if (pred(i) > max) {
            argmax = i;
            max = pred(i);
        }

        if (pred(i) < min) {
            argmin = i;
            min = pred(i);
        }
    }

    if (ebt::in(std::string("print-attention"), args)) {
        la::vector<double>& u = autodiff::get_output<la::vector<double>>(atts[argmax]);
        std::cout << id_label[argmax] << " ";
        std::cout << std::vector<double> { u.data(), u.data() + u.size() } << std::endl;

        la::vector<double>& v = autodiff::get_output<la::vector<double>>(atts[argmin]);
        std::cout << id_label[argmin] << " ";
        std::cout << std::vector<double> { v.data(), v.data() + v.size() } << std::endl;

        std::cout << std::endl;
    } else {
        std::cout << id_label[argmax] << std::endl;
    }
}

