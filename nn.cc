#include "nn/nn.h"
#include "opt/opt.h"
#include <fstream>

#if USE_GPU
#include "la/la-gpu.h"
#endif

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

    void iadd(param_t& p, param_t const& q)
    {
        for (int i = 0; i < p.weight.size(); ++i) {
	    la::iadd(p.weight[i], q.weight[i]);
            la::iadd(p.bias[i], q.bias[i]);
        }
	la::iadd(p.label_weight, q.label_weight);
        la::iadd(p.label_bias, q.label_bias);
    }

    void resize_as(param_t& p, param_t const& q)
    {
        for (int i = 0; i < q.weight.size(); ++i) {
            la::matrix<double> m;
            m.resize(q.weight[i].rows(), q.weight[i].cols());
            p.weight.push_back(std::move(m));
        }

        for (int i = 0; i < q.bias.size(); ++i) {
            la::vector<double> v;
            v.resize(q.bias[i].size());
            p.bias.push_back(std::move(v));
        }

        p.label_weight.resize(q.label_weight.rows(), q.label_weight.cols());

        p.label_bias.resize(q.label_bias.size());
    }

    opt_t load_opt(std::istream& is)
    {
        opt_t o;
        std::string line;
        std::getline(is, line);
        o.time = std::stoi(line);
        o.first_moment = load_param(is);
        o.second_moment = load_param(is);
        return o;
    }

    opt_t load_opt(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_opt(ifs);
    }

    void save_opt(opt_t const& o, std::ostream& os)
    {
        os << o.time << std::endl;
        save_param(o.first_moment, os);
        save_param(o.second_moment, os);
    }

    void save_opt(opt_t const& o, std::string filename)
    {
        std::ofstream ofs { filename };
        save_opt(o, ofs);
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
            auto b_var = nn.graph.var(p.bias[i]);
            nn.weight.push_back(w_var);
            nn.bias.push_back(b_var);
            nn.hidden.push_back(autodiff::relu(
                autodiff::add(autodiff::mul(w_var, nn.hidden.back()), b_var)
            ));
        }

        nn.label_weight = nn.graph.var(p.label_weight);
        nn.label_bias = nn.graph.var(p.label_bias);

        nn.output = autodiff::logsoftmax(autodiff::add(
            autodiff::mul(nn.label_weight, nn.hidden.back()), nn.label_bias));

        return nn;
    }

    nn_t make_nn2(param_t const& p)
    {
        nn_t nn;

        nn.hidden.push_back(nn.graph.var());

        for (int i = 0; i < p.weight.size(); ++i) {
            auto w_var = nn.graph.var(p.weight[i]);
            auto b_var = nn.graph.var(p.bias[i]);
            nn.weight.push_back(w_var);
            nn.bias.push_back(b_var);
            nn.hidden.push_back(autodiff::relu(
                autodiff::add(autodiff::mul(w_var, nn.hidden.back()), b_var)
            ));
        }

        nn.label_weight = nn.graph.var(p.label_weight);
        nn.label_bias = nn.graph.var(p.label_bias);

        std::vector<std::shared_ptr<autodiff::op_t>> hiddens {nn.hidden.begin() + 1, nn.hidden.end()};

        nn.output = autodiff::logsoftmax(autodiff::add(
            autodiff::mul(nn.label_weight, autodiff::add(hiddens)), nn.label_bias));

        return nn;
    }

    void adagrad_update(param_t& p, param_t const& grad, opt_t& opt_data, double step_size)
    {
        for (int i = 0; i < p.weight.size(); ++i) {
            opt::adagrad_update(p.weight[i], grad.weight[i], opt_data.first_moment.weight[i], step_size);
            opt::adagrad_update(p.bias[i], grad.bias[i], opt_data.first_moment.bias[i], step_size);
        }

        opt::adagrad_update(p.label_weight, grad.label_weight,
            opt_data.first_moment.label_weight, step_size);
        opt::adagrad_update(p.label_bias, grad.label_bias,
            opt_data.first_moment.label_bias, step_size);
    }

    void adam_update(param_t& p, param_t const& grad, opt_t& opt_data, double step_size)
    {
        ++opt_data.time;

        for (int i = 0; i < p.weight.size(); ++i) {
            opt::adam_update(p.weight[i], grad.weight[i],
                opt_data.first_moment.weight[i], opt_data.second_moment.weight[i],
                1 + opt_data.time / 1e6, step_size, 0.9, 0.999);
            opt::adam_update(p.bias[i], grad.bias[i],
                opt_data.first_moment.bias[i], opt_data.second_moment.bias[i],
                1 + opt_data.time / 1e6, step_size, 0.9, 0.999);
        }

        opt::adam_update(p.label_weight, grad.label_weight,
            opt_data.first_moment.label_weight, opt_data.second_moment.label_weight,
            1 + opt_data.time / 1e6, step_size, 0.9, 0.999);
        opt::adam_update(p.label_bias, grad.label_bias,
            opt_data.first_moment.label_bias, opt_data.second_moment.label_bias,
            1 + opt_data.time / 1e6, step_size, 0.9, 0.999);
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
        return la::mul(gold, -1);
    }
}

