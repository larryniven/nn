#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "opt/opt.h"
#include "nn/rnn.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    lstm::dblstm_param_t param;
    lstm::dblstm_param_t opt_data;
    lstm::dblstm_nn_t nn;

    double step_size;

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
        "learn-rnn",
        "Train a LSTM frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
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

    param = lstm::load_dblstm_param(args.at("param"));
    opt_data = lstm::load_dblstm_param(args.at("opt-data"));

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    step_size = std::stod(args.at("step-size"));

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

        nn = lstm::make_dblstm_nn(param, frames);

        lstm::eval(nn);

        double loss_sum = 0;
        double nframes = 0;

        for (int t = 0; t < nn.logprob.size(); ++t) {
            auto& pred = autodiff::get_output<la::vector<double>>(nn.logprob.at(t));
            la::vector<double> gold;
            gold.resize(label_id.size());
            gold(label_id.at(labels[t])) = 1;
            lstm::log_loss loss { gold, pred };
            nn.logprob[t]->grad = std::make_shared<la::vector<double>>(loss.grad());
            if (std::isnan(loss.loss())) {
                std::cerr << "loss is nan" << std::endl;
                exit(1);
            } else {
                loss_sum += loss.loss();
                nframes += 1;
            }
        }

        std::cout << "loss: " << loss_sum / nframes << std::endl;

        lstm::grad(nn);

        lstm::dblstm_param_t grad = lstm::copy_dblstm_grad(nn);

        lstm::adagrad_update(param, grad, opt_data, step_size);

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

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    lstm::save_dblstm_param(param, output_param);
    lstm::save_dblstm_param(opt_data, output_opt_data);
}

