#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    lstm::dblstm_param_t param;
    lstm::dblstm_nn_t nn;

    std::vector<std::string> label;

    double rnndrop_prob;

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
            {"logprob", "", false}
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

    param = lstm::load_dblstm_param(args.at("param"));

    label = speech::load_label_set(args.at("label"));

    if (ebt::in(std::string("rnndrop-prob"), args)) {
        rnndrop_prob = std::stod(args.at("rnndrop-prob"));
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

        nn = make_dblstm_nn(param, frames);

        if (ebt::in(std::string("rnndrop-prob"), args)) {
            for (int ell = 1; ell < nn.layer.size(); ++ell) {
                la::vector<double> mask_vec;
                mask_vec.resize(param.layer[ell].forward_param.hidden_input.cols(), rnndrop_prob);

                auto& f_mask = nn.layer[ell].forward_feat_nn.input_mask;
                f_mask->output = std::make_shared<la::vector<double>>(mask_vec);

                auto& b_mask = nn.layer[ell].backward_feat_nn.input_mask;
                b_mask->output = std::make_shared<la::vector<double>>(mask_vec);
            }
        }

        lstm::eval(nn);

        std::cout << i << ".phn" << std::endl;

        if (ebt::in(std::string("logprob"), args)) {
            for (int t = 0; t < nn.logprob.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(nn.logprob.at(t));

                std::cout << pred(0);

                for (int j = 1; j < pred.size(); ++j) {
                    std::cout << " " << pred(j);
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < nn.logprob.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(nn.logprob.at(t));

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

