#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "opt/opt.h"
#include "nn/lstm-gpu.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    lstm::gpu::dblstm_param_t param;
    lstm::gpu::dblstm_param_t opt_data;
    lstm::gpu::dblstm_nn_t nn;

    double step_size;
    double momentum;

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
        "learn-lstm-gpu",
        "Train a LSTM frame classifier",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"momentum", "", false},
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

    param = lstm::gpu::to_device(
        lstm::load_dblstm_param(args.at("param")));
    opt_data = lstm::gpu::to_device(
        lstm::load_dblstm_param(args.at("opt-data")));

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("momentum"), args)) {
        momentum = std::stod(args.at("momentum"));
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

    lstm::gpu::dblstm_param_t grad;
    lstm::gpu::resize_as(grad, param);

    autodiff::gpu::memory_pool<double> mem { 100000000 };

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> labels;

        labels = speech::load_label_batch(label_batch);

        if (!frame_batch || !label_batch) {
            break;
        }

        nn = lstm::gpu::make_dblstm_nn(param, mem, frames);

        lstm::gpu::eval(nn);

        double loss_sum = 0;
        double nframes = 0;

        std::vector<double> gold_block;

        for (int t = 0; t < nn.logprob.size(); ++t) {
            std::vector<double> gold;
            gold.resize(label_id.size());
            gold[label_id.at(labels[t])] = 1;

            gold_block.insert(gold_block.end(), gold.begin(), gold.end());
        }

        double *d = mem.alloc(gold_block.size());
        la::gpu::weak_vector<double> gold_device_block { d, gold_block.size() };
        la::gpu::to_device(gold_device_block, la::vector<double>(gold_block));
        unsigned int gold_dim = label_id.size();

        for (int t = 0; t < nn.logprob.size(); ++t) {
            auto& pred = autodiff::get_output<la::gpu::vector_like<double>>(nn.logprob.at(t));

            lstm::gpu::log_loss loss {
                la::gpu::weak_vector<double>(gold_device_block.data() + t * gold_dim, gold_dim),
                pred };

            if (std::isnan(loss.loss())) {
                std::cerr << "loss is nan" << std::endl;
                exit(1);
            } else {
                loss_sum += loss.loss();
                nframes += 1;
            }

            la::gpu::weak_vector<double> g(gold_device_block.data() + t * gold_dim, gold_dim);

            la::gpu::imul(g, -1);

            nn.logprob[t]->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
        }


        lstm::gpu::attach_grad(grad, nn);
        lstm::gpu::grad(nn);

        if (ebt::in(std::string("momentum"), args)) {
            lstm::gpu::const_step_update_momentum(param, grad, opt_data, momentum, step_size);
        } else {
            lstm::gpu::adagrad_update(param, grad, opt_data, step_size);
        }

        lstm::gpu::zero(grad);

        mem.reset();

        std::cout << "loss: " << loss_sum / nframes << std::endl;

#if 0
        {
            lstm::param_t p = param;
            p.hidden_input(0, 0) += 1e-8;
            lstm::nn_t nn2 = lstm::make_nn(p, frames);
            lstm::eval(nn2);
            auto& pred = autodiff::get_output<la::vector_like<double>>(nn2.logprob.at(1));
            la::vector<double> gold;
            gold.resize(label_id.size());
            gold(label_id.at(labels[1])) = 1;
            lstm::log_loss loss2 { gold, pred };

            auto& grad = autodiff::get_grad<la::matrix_like<double>>(nn.hidden_input);
            std::cout << (loss2.loss() - loss_1) / 1e-8 << " " << grad(0, 0) << std::endl;
        }
#endif

#ifdef DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    lstm::save_dblstm_param(lstm::gpu::to_host(param), output_param);
    lstm::save_dblstm_param(lstm::gpu::to_host(opt_data), output_opt_data);
}

