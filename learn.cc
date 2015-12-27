#include "autodiff/autodiff.h"
#include "la/la.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace nn {

    struct nn_t {
        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> weight;
        std::vector<std::shared_ptr<autodiff::op_t>> bias;
        std::shared_ptr<autodiff::op_t> label_weight;
        std::shared_ptr<autodiff::op_t> label_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> hidden;
        std::shared_ptr<autodiff::op_t> output;
    };
    
    struct param_t {
        std::vector<la::matrix<double>> weight;
        std::vector<la::vector<double>> bias;
        la::matrix<double> label_weight;
        la::vector<double> label_bias;
    };
    
    param_t load_param(std::istream& is);
    param_t load_param(std::string filename);

    void save_param(param_t const& p, std::ostream& os);
    void save_param(param_t const& p, std::string filename);

    std::pair<
        std::unordered_map<std::string, int>,
        std::vector<std::string>
    >
    load_label_map(std::string filename);

    nn_t make_nn(param_t const& p);

    void adagrad_update(param_t& p, param_t const& grad, param_t& opt_data, double step_size);

    param_t copy_grad(nn_t const& nn);

}

struct learning_env {
    std::ifstream input_list;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> labels;

    nn::param_t param;
    nn::param_t opt_data;

    double step_size;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();
};

struct log_loss {

    la::vector<double> pred;
    la::vector<double> gold;

    log_loss(la::vector<double> const& pred,
        la::vector<double> const& gold);

    double loss();

    la::vector<double> grad();

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

    param = nn::load_param(args.at("param"));
    opt_data = nn::load_param(args.at("opt-data"));

    step_size = std::stod(args.at("step-size"));

    output_param = args.at("output-param");
    output_opt_data = args.at("output-opt-data");
}

void learning_env::run()
{
    std::string line;

    int sample = 0;
    double accu_loss = 0;
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

        nn::nn_t nn = nn::make_nn(param);

        nn.hidden[0]->output = std::make_shared<la::vector<double>>(la::vector<double>(input));

        autodiff::eval(nn.output, autodiff::eval_funcs);
        log_loss loss { autodiff::get_output<la::vector<double>>(nn.output), gold };

        accu_loss += loss.loss();

#if DEBUG
        {
            auto& tmp = param.weight[0](0, 1);
            tmp += 1e-8;

            nn::nn_t nn = nn::make_nn(param);

            nn.hidden[0]->output = std::make_shared<la::vector<double>>(la::vector<double>(input));

            autodiff::eval(nn.output, autodiff::eval_funcs);
            log_loss loss2 { autodiff::get_output<la::vector<double>>(nn.output), gold };

            std::cout << "numerical grad: " << (loss2.loss() - loss.loss()) / 1e-8 << std::endl;
        }
#endif

        nn.output->grad = std::make_shared<la::vector<double>>(loss.grad());
        autodiff::grad(nn.output, autodiff::grad_funcs);

#if DEBUG
        std::cout << "calc grad: " << autodiff::get_grad<la::matrix<double>>(nn.weight[0])(0, 1) << std::endl;
#endif

        nn::param_t grad = nn::copy_grad(nn);

        nn::adagrad_update(param, grad, opt_data, step_size);

        if (sample % 100 == 0) {
            std::cout << "last 100 avg loss: " << accu_loss / 100 << std::endl;
            accu_loss = 0;
        }

        ++sample;
    }

    save_param(param, output_param);
    save_param(opt_data, output_opt_data);
}

namespace nn {

    param_t load_param(std::istream& is)
    {
        param_t p;
        std::string line;

        ebt::json::json_parser<decltype(p.weight)> weight_parser;
        p.weight = weight_parser.parse(is);

        std::getline(is, line);

        ebt::json::json_parser<decltype(p.bias)> bias_parser;
        p.bias = bias_parser.parse(is);

        std::getline(is, line);

        ebt::json::json_parser<decltype(p.label_weight)> label_weight_parser;
        p.label_weight = label_weight_parser.parse(is);

        std::getline(is, line);

        ebt::json::json_parser<decltype(p.label_bias)> label_bias_parser;
        p.label_bias = label_bias_parser.parse(is);

        std::getline(is, line);

        return p;
    }

    param_t load_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_param(ifs);
    }

    void save_param(param_t const& p, std::ostream& os)
    {
        using ebt::json::dump;

        dump(p.weight, os);
        os << std::endl;
        dump(p.bias, os);
        os << std::endl;
        dump(p.label_weight, os);
        os << std::endl;
        dump(p.label_bias, os);
        os << std::endl;
    }

    void save_param(param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_param(p, ofs);
    }

    std::pair<
        std::unordered_map<std::string, int>,
        std::vector<std::string>
    >
    load_label_map(std::string filename)
    {
        std::ifstream ifs { filename };

        std::unordered_map<std::string, int> label_id;
        std::vector<std::string> label;

        std::string line;

        int i = 0;
        while (std::getline(ifs, line)) {
            label_id[line] = i;
            label.push_back(line);
            ++i;
        }

        return std::make_pair(label_id, label);
    }

    nn_t make_nn(param_t const& p)
    {
        nn_t nn;

        nn.hidden.push_back(nn.graph.var());

        for (int i = 0; i < p.weight.size(); ++i) {
            auto w_var = nn.graph.var(p.weight[i]);
            nn.weight.push_back(w_var);
            auto b_var = nn.graph.var(p.bias[i]);
            nn.bias.push_back(b_var);

            nn.hidden.push_back(autodiff::logistic(
                autodiff::add(autodiff::mult(w_var, nn.hidden.back()), b_var)
            ));
        }

        nn.label_weight = nn.graph.var(p.label_weight);
        nn.label_bias = nn.graph.var(p.label_bias);

        nn.output = autodiff::logsoftmax(autodiff::add(
            autodiff::mult(nn.label_weight, nn.hidden.back()), nn.label_bias));

        return nn;
    }

    void adagrad_update(param_t& p, param_t const& grad, param_t& opt_data, double step_size)
    {
        for (int i = 0; i < p.weight.size(); ++i) {
            opt::adagrad_update(p.weight[i], grad.weight[i], opt_data.weight[i], step_size);
            opt::adagrad_update(p.bias[i], grad.bias[i], opt_data.bias[i], step_size);
        }

        opt::adagrad_update(p.label_weight, grad.label_weight,
            opt_data.label_weight, step_size);
        opt::adagrad_update(p.label_bias, grad.label_bias,
            opt_data.label_bias, step_size);
    }

    param_t copy_grad(nn_t const& nn)
    {
        param_t result;

        for (int i = 0; i < nn.weight.size(); ++i) {
            result.weight.push_back(autodiff::get_grad<la::matrix<double>>(nn.weight[i]));
            result.bias.push_back(autodiff::get_grad<la::vector<double>>(nn.bias[i]));
        }

        result.label_weight = autodiff::get_grad<la::matrix<double>>(nn.label_weight);
        result.label_bias = autodiff::get_grad<la::vector<double>>(nn.label_bias);

        return result;
    }

}

log_loss::log_loss(
    la::vector<double> const& pred, la::vector<double> const& gold)
    : pred(pred), gold(gold)
{}

double log_loss::loss()
{
    return -la::dot(pred, gold);
}

la::vector<double> log_loss::grad()
{
    return la::mult(gold, -1);
}
