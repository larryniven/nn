#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/gru.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    gru::dbgru_feat_param_t dbgru_param;
    gru::pred_param_t pred_param;

    gru::dbgru_feat_nn_t dbgru_nn;
    gru::pred_nn_t pred_nn;

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

    std::ifstream ifs { args.at("param") };
    dbgru_param = gru::load_dbgru_feat_param(ifs);
    pred_param = gru::load_pred_param(ifs);
    ifs.close();

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

        autodiff::computation_graph g;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;
        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(g.var(la::vector<double>(frames[i])));
        }

        gru::dbgru_feat_nn_t dbgru_feat_nn = gru::make_dbgru_feat_nn(g, dbgru_param, inputs);
        gru::pred_nn_t pred_nn = gru::make_pred_nn(g, pred_param, dbgru_feat_nn.layer.back().output);

        gru::eval(pred_nn);

        std::cout << i << ".phn" << std::endl;

        for (int t = 0; t < pred_nn.logprob.size(); ++t) {
            auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob.at(t));

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

        std::cout << "." << std::endl;

        ++i;
    }
}

