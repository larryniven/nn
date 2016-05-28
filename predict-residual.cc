#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/residual.h"
#include "nn/pred.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    residual::nn_param_t param;
    residual::nn_t nn;
    nn::pred_param_t pred_param;
    nn::pred_nn_t pred_nn;

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
    param = residual::load_nn_param(param_ifs);
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

        std::cout << i << ".phn" << std::endl;

        for (int t = 0; t < frames.size(); ++t) {
            autodiff::computation_graph graph;

            nn = residual::make_nn(graph, param);
            pred_nn = nn::make_pred_nn(graph, nn.layer.back().output, pred_param);

            std::vector<double> input;

            for (int k = -5; k <= 5; ++k) {
                if (0 <= k + t && k + t < frames.size()) {
                    input.insert(input.end(), frames[k + t].begin(), frames[k + t].end());
                } else {
                    input.resize(input.size() + frames.front().size());
                }
            }

            nn.input->output = std::make_shared<la::vector<double>>(la::vector<double> { std::move(input) });
            autodiff::eval(pred_nn.logprob, autodiff::eval_funcs);

            if (ebt::in(std::string("logprob"), args)) {
                auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob);

                std::cout << pred(0);
                for (int j = 1; j < pred.size(); ++j) {
                    std::cout << " " << pred(j);
                }
                std::cout << std::endl;
            } else {
                auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob);

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

