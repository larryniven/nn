#include "autodiff/autodiff.h"
#include "la/la.h"
#include "opt/opt.h"
#include "nn/nn.h"
#include <fstream>
#include <algorithm>

struct prediction_env {
    std::ifstream input_list;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> labels;

    nn::param_t param;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();
};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict",
        "Predict labels with a feed-forward neural network",
        {
            {"input-list", "", true},
            {"label", "", true},
            {"param", "", true},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    std::unordered_map<std::string, std::string> args = ebt::parse_args(argc, argv, spec);

    prediction_env env { args };
    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_list.open(args.at("input-list"));

    std::tie(label_id, labels) = nn::load_label_map(args.at("label"));

    param = nn::load_param(args.at("param"));
}

void prediction_env::run()
{
    std::string line;

    int sample = 0;
    while (std::getline(input_list, line)) {
        std::vector<std::string> parts = ebt::split(line);

        std::vector<double> input;
        input.resize(parts.size());
        std::transform(parts.begin(), parts.end(), input.begin(),
            [](std::string const& s) { return std::stod(s); });

        nn::nn_t nn = nn::make_nn(param);

        nn.hidden[0]->output = std::make_shared<la::vector<double>>(la::vector<double>(input));

        autodiff::eval(nn.output, autodiff::eval_funcs);

        auto& logp = autodiff::get_output<la::vector<double>>(nn.output);

        double max = -std::numeric_limits<double>::infinity();
        int argmax = -1;
        for (int i = 0; i < logp.size(); ++i) {
            if (logp(i) > max) {
                max = logp(i);
                argmax = i;
            }
        }

        std::cout << labels[argmax] << std::endl;

        ++sample;
    }
}
