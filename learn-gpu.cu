#include <fstream>
#include <algorithm>
#include "ebt/ebt.h"
#include "autodiff/autodiff.h"
#include "la/la.h"
#include "opt/opt.h"
#include "nn/nn.h"

#include "la/la-gpu.h"
#include "autodiff/autodiff-gpu.h"
#include "nn/nn-gpu.h"

#if 0
#include <cuda_profiler_api.h>
#endif

struct learning_env {
    std::ifstream input_list;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> labels;

    nn::gpu::param_t param;
    nn::gpu::opt_t opt_data;

    double step_size;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();
};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Train a feed-forward neural network",
        {
            {"input-list", "", true},
            {"label", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"output-param", "", true},
            {"output-opt-data", "", true}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    std::unordered_map<std::string, std::string> args = ebt::parse_args(argc, argv, spec);

    learning_env env { args };
    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_list.open(args.at("input-list"));

    std::tie(label_id, labels) = nn::load_label_map(args.at("label"));

    param = nn::gpu::param_t(nn::load_param(args.at("param")));
    opt_data = nn::gpu::opt_t(nn::load_opt(args.at("opt-data")));

    step_size = std::stod(args.at("step-size"));

    output_param = args.at("output-param");
    output_opt_data = args.at("output-opt-data");
}

void learning_env::run()
{
    std::string line;

    int sample = 0;
    double loss_mean = 0;
    double loss_var = 0;

    nn::nn_t nn = nn::gpu::make_nn(param);

    while (std::getline(input_list, line)) {
        std::vector<std::string> parts = ebt::split(line);

        std::vector<double> input;
        input.resize(parts.size() - 1);

        std::transform(parts.begin() + 1, parts.end(), input.begin(),
            [](std::string const& s) { return std::stod(s); });

        std::string label = parts.front();

        la::vector<double> gold;
        gold.resize(label_id.size());
        gold(label_id[label]) = 1;

        if (nn.hidden[0]->output == nullptr) {
            nn.hidden[0]->output = std::make_shared<la::gpu::vector<double>>(
                la::gpu::vector<double>(la::vector<double>(input)));
        } else {
            auto& v = autodiff::get_output<la::gpu::vector_like<double>>(nn.hidden[0]);
            la::gpu::to_device(v, la::vector<double>(input));
        }
        autodiff::eval(nn.output, autodiff::gpu::eval_funcs);
        nn::gpu::log_loss loss { autodiff::get_output<la::gpu::vector<double>>(nn.output),
            la::gpu::vector<double>(gold) };

        loss_mean += loss.loss();
        loss_var += std::pow(loss.loss(), 2);

#if DEBUG
        {
            auto& tmp = param.weight[0](0, 0);
            double backup = tmp;
            tmp += 1e-8;

            nn::nn_t nn = nn::make_nn(param);

            nn.hidden[0]->output = std::make_shared<la::gpu::vector<double>>(
                la::gpu::vector<double>(la::vector<double>(input)));
            autodiff::eval(nn.output, autodiff::gpu::eval_funcs);
            nn::gpu::log_loss loss2 { autodiff::get_output<la::gpu::vector<double>>(nn.output), gold };

            std::cout << "numerical grad: " << (loss2.loss() - loss.loss()) / 1e-8 << std::endl;
            tmp = backup;
        }
#endif

        nn.output->grad = std::make_shared<la::gpu::vector<double>>(
            la::gpu::vector<double>(loss.grad()));
        autodiff::grad(nn.output, autodiff::gpu::grad_funcs);

#if DEBUG
        std::cout << "calc grad: "
            << to_host(autodiff::get_grad<la::gpu::matrix<double>>(nn.weight[0]))(0, 0)
            << std::endl;
#endif

        nn::gpu::param_t grad;
        grad.weight.resize(nn.weight.size());
        grad.bias.resize(nn.bias.size());
        nn::gpu::move_grad(grad, nn);
        nn::gpu::move_param(param, nn);

        nn::gpu::adagrad_update(param, grad, opt_data, step_size);

        nn::gpu::move_grad(nn, grad);
        nn::gpu::move_param(nn, param);
        nn::gpu::zero_grad(nn);

        if (sample % 100 == 0) {
            std::cout << "last 100 avg loss: " << loss_mean / 100
                << " var: " << loss_var / 100 - std::pow(loss_mean / 100, 2) << std::endl;
            loss_mean = 0;
            loss_var = 0;
        }

        ++sample;
    }

    nn::gpu::move_param(param, nn);
    save_param(nn::gpu::to_host(param), output_param);
    save_opt(nn::gpu::to_host(opt_data), output_opt_data);
}
