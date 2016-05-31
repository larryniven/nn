#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/pred.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    lstm::db_lstm2d_param_t param;
    lstm::db_lstm2d_nn_t nn;
    nn::pred_param_t pred_param;
    rnn::pred_nn_t pred_nn;

    std::vector<std::string> label;

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
            {"logprob", "", false},
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
    param = lstm::load_db_lstm2d_param(param_ifs);
    pred_param = nn::load_pred_param(param_ifs);
    param_ifs.close();

    label = speech::load_label_set(args.at("label"));
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

        nn = lstm::make_db_lstm2d_nn(graph, param, inputs);

        pred_nn = rnn::make_pred_nn(graph, pred_param, nn.layer.back().output);

        auto topo_order = autodiff::topo_order(pred_nn.logprob);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        std::cout << i << ".phn" << std::endl;

        if (ebt::in(std::string("logprob"), args)) {
            for (int t = 0; t < pred_nn.logprob.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob[t]);

                std::cout << pred(0);

                for (int j = 1; j < pred.size(); ++j) {
                    std::cout << " " << pred(j);
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < pred_nn.logprob.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob[t]);

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

