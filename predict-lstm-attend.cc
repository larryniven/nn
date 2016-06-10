#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/pred.h"
#include <fstream>
#include "nn/attention.h"

struct prediction_env {

    std::ifstream frame_batch;

    lstm::dblstm_feat_param_t param;
    lstm::dblstm_feat_nn_t nn;
    nn::pred_param_t pred_param;
    rnn::pred_nn_t pred_nn;

    std::vector<std::string> label;

    double rnndrop_prob;
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
            {"rnndrop-prob", "", false},
            {"logprob", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
            {"attention", "", false}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    std::ifstream param_ifs { args.at("param") };
    param = lstm::load_dblstm_feat_param(param_ifs);
    pred_param = nn::load_pred_param(param_ifs);
    param_ifs.close();

    label = speech::load_label_set(args.at("label"));

    if (ebt::in(std::string("rnndrop-prob"), args)) {
        rnndrop_prob = std::stod(args.at("rnndrop-prob"));
    }

    subsample_freq = 1;
    if (ebt::in(std::string("subsample-freq"), args)) {
        subsample_freq = std::stoi(args.at("subsample-freq"));
    }

    subsample_shift = 0;
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

        std::vector<std::shared_ptr<autodiff::op_t>> subsampled_inputs
            = rnn::subsample_input(inputs, subsample_freq, subsample_shift);

        nn = lstm::make_dblstm_feat_nn(graph, param, subsampled_inputs);

        if (ebt::in(std::string("rnndrop-prob"), args)) {
            lstm::apply_mask(nn, param, rnndrop_prob);
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

        std::cout << i << ".phn" << std::endl;

        if (ebt::in(std::string("logprob"), args)) {
            for (int t = 0; t < upsampled_output.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(upsampled_output[t]);

                std::cout << pred(0);

                for (int j = 1; j < pred.size(); ++j) {
                    std::cout << " " << pred(j);
                }

                std::cout << std::endl;
            }
        } else if (ebt::in(std::string("attention"), args)) {
            for (int i = 0; i < atts.size(); ++i) {
                auto& pred = autodiff::get_output<la::vector<double>>(upsampled_output[i]);

                int argmax = -1;
                double max = -std::numeric_limits<double>::infinity();

                for (int j = 0; j < pred.size(); ++j) {
                    if (pred(j) > max) {
                        max = pred(j);
                        argmax = j;
                    }
                }

                std::cout << label[argmax] << " " << i << " ";

                auto& v = autodiff::get_output<la::vector<double>>(atts[i].attention);
                std::cout << "[";
                for (int j = 0; j < v.size(); ++j) {
                    if (j == i) {
                        std::cout << "*";
                    }
                    std::cout << v(j);
                    if (j != v.size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]";
                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < upsampled_output.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(upsampled_output[t]);

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

        ++i;
    }
}

