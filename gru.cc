#include "nn/gru.h"
#include <fstream>
#include "opt/opt.h"
#include <algorithm>

namespace gru {

    gru_feat_param_t load_gru_feat_param(std::istream& is)
    {
        ebt::json::json_parser<la::matrix<double>> mat_parser;
        ebt::json::json_parser<la::vector<double>> vec_parser;

        gru_feat_param_t result;
        std::string line;

        result.reset_input = mat_parser.parse(is);
        std::getline(is, line);
        result.reset_hidden = mat_parser.parse(is);
        std::getline(is, line);
        result.reset_bias = vec_parser.parse(is);
        std::getline(is, line);

        result.update_input = mat_parser.parse(is);
        std::getline(is, line);
        result.update_hidden = mat_parser.parse(is);
        std::getline(is, line);
        result.update_bias = vec_parser.parse(is);
        std::getline(is, line);

        result.candidate_input = mat_parser.parse(is);
        std::getline(is, line);
        result.candidate_hidden = mat_parser.parse(is);
        std::getline(is, line);
        result.candidate_bias = vec_parser.parse(is);
        std::getline(is, line);

        result.shortcut_input = mat_parser.parse(is);
        std::getline(is, line);
        result.shortcut_bias = vec_parser.parse(is);
        std::getline(is, line);

        return result;
    }

    gru_feat_param_t load_gru_feat_param(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_gru_feat_param(ifs);
    }

    void save_gru_feat_param(gru_feat_param_t const& param, std::ostream& os)
    {
        ebt::json::dump(param.reset_input, os);
        os << std::endl;
        ebt::json::dump(param.reset_hidden, os);
        os << std::endl;
        ebt::json::dump(param.reset_bias, os);
        os << std::endl;

        ebt::json::dump(param.update_input, os);
        os << std::endl;
        ebt::json::dump(param.update_hidden, os);
        os << std::endl;
        ebt::json::dump(param.update_bias, os);
        os << std::endl;

        ebt::json::dump(param.candidate_input, os);
        os << std::endl;
        ebt::json::dump(param.candidate_hidden, os);
        os << std::endl;
        ebt::json::dump(param.candidate_bias, os);
        os << std::endl;

        ebt::json::dump(param.shortcut_input, os);
        os << std::endl;
        ebt::json::dump(param.shortcut_bias, os);
        os << std::endl;
    }

    void save_gru_feat_param(gru_feat_param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };

        save_gru_feat_param(param, ofs);
    }

    void adagrad_update(gru_feat_param_t& param, gru_feat_param_t const& grad,
        gru_feat_param_t& opt_data, double step_size)
    {
        opt::adagrad_update(param.reset_input, grad.reset_input,
            opt_data.reset_input, step_size);
        opt::adagrad_update(param.reset_hidden, grad.reset_hidden,
            opt_data.reset_hidden, step_size);
        opt::adagrad_update(param.reset_bias, grad.reset_bias,
            opt_data.reset_bias, step_size);

        opt::adagrad_update(param.update_input, grad.update_input,
            opt_data.update_input, step_size);
        opt::adagrad_update(param.update_hidden, grad.update_hidden,
            opt_data.update_hidden, step_size);
        opt::adagrad_update(param.update_bias, grad.update_bias,
            opt_data.update_bias, step_size);

        opt::adagrad_update(param.candidate_input, grad.candidate_input,
            opt_data.candidate_input, step_size);
        opt::adagrad_update(param.candidate_hidden, grad.candidate_hidden,
            opt_data.candidate_hidden, step_size);
        opt::adagrad_update(param.candidate_bias, grad.candidate_bias,
            opt_data.candidate_bias, step_size);

        opt::adagrad_update(param.shortcut_input, grad.shortcut_input,
            opt_data.shortcut_input, step_size);
        opt::adagrad_update(param.shortcut_bias, grad.shortcut_bias,
            opt_data.shortcut_bias, step_size);
    }

    void rmsprop_update(gru_feat_param_t& param, gru_feat_param_t const& grad,
        gru_feat_param_t& opt_data, double decay, double step_size)
    {
        opt::rmsprop_update(param.reset_input, grad.reset_input,
            opt_data.reset_input, decay, step_size);
        opt::rmsprop_update(param.reset_hidden, grad.reset_hidden,
            opt_data.reset_hidden, decay, step_size);
        opt::rmsprop_update(param.reset_bias, grad.reset_bias,
            opt_data.reset_bias, decay, step_size);

        opt::rmsprop_update(param.update_input, grad.update_input,
            opt_data.update_input, decay, step_size);
        opt::rmsprop_update(param.update_hidden, grad.update_hidden,
            opt_data.update_hidden, decay, step_size);
        opt::rmsprop_update(param.update_bias, grad.update_bias,
            opt_data.update_bias, decay, step_size);

        opt::rmsprop_update(param.candidate_input, grad.candidate_input,
            opt_data.candidate_input, decay, step_size);
        opt::rmsprop_update(param.candidate_hidden, grad.candidate_hidden,
            opt_data.candidate_hidden, decay, step_size);
        opt::rmsprop_update(param.candidate_bias, grad.candidate_bias,
            opt_data.candidate_bias, decay, step_size);

        opt::rmsprop_update(param.shortcut_input, grad.shortcut_input,
            opt_data.shortcut_input, decay, step_size);
        opt::rmsprop_update(param.shortcut_bias, grad.shortcut_bias,
            opt_data.shortcut_bias, decay, step_size);
    }

    gru_feat_nn_t make_gru_feat_nn(autodiff::computation_graph& g,
        gru_feat_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        gru_feat_nn_t result;

        la::vector<double> one;
        one.resize(param.candidate_input.rows());
        result.one = g.var(one);

        result.reset_input = g.var(param.reset_input);
        result.reset_hidden = g.var(param.reset_hidden);
        result.reset_bias = g.var(param.reset_bias);

        result.update_input = g.var(param.update_input);
        result.update_hidden = g.var(param.update_hidden);
        result.update_bias = g.var(param.update_bias);

        result.candidate_input = g.var(param.candidate_input);
        result.candidate_hidden = g.var(param.candidate_hidden);
        result.candidate_bias = g.var(param.candidate_bias);

        result.shortcut_input = g.var(param.shortcut_input);
        result.shortcut_bias = g.var(param.shortcut_bias);

        result.candidate.push_back(autodiff::add(
            autodiff::tanh(autodiff::add(
                autodiff::mul(result.candidate_input, inputs.front()),
                result.candidate_bias)),
            autodiff::add(
                autodiff::mul(result.shortcut_input, inputs.front()),
                result.shortcut_bias)
        ));

        result.update.push_back(autodiff::logistic(autodiff::add(
            autodiff::mul(result.update_input, inputs.front()),
            result.update_bias)));

        result.hidden.push_back(autodiff::emul(result.update.back(), result.candidate.back()));

        for (int i = 1; i < inputs.size(); ++i) {
            result.reset.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.reset_input, inputs[i]),
                    autodiff::mul(result.reset_hidden, result.hidden.back()),
                    result.reset_bias
                })));

            result.candidate.push_back(autodiff::add(
                autodiff::tanh(autodiff::add(
                    std::vector<std::shared_ptr<autodiff::op_t>> {
                        autodiff::mul(result.candidate_input, inputs[i]),
                        autodiff::mul(result.candidate_hidden,
                            autodiff::emul(result.reset.back(), result.hidden.back())),
                        result.candidate_bias
                    })),
                autodiff::add(
                    autodiff::mul(result.shortcut_input, inputs[i]),
                    result.shortcut_bias)
                ));

            result.update.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.update_input, inputs[i]),
                    autodiff::mul(result.update_hidden, result.hidden.back()),
                    result.update_bias
                })));

            result.hidden.push_back(autodiff::add(
                autodiff::emul(autodiff::sub(result.one, result.update.back()), result.hidden.back()),
                autodiff::emul(result.update.back(), result.candidate.back())
            ));
        }

        return result;
    }

    gru_feat_param_t copy_grad(gru_feat_nn_t const& nn)
    {
        gru_feat_param_t result;

        result.reset_input = autodiff::get_grad<la::matrix<double>>(nn.reset_input);
        result.reset_hidden = autodiff::get_grad<la::matrix<double>>(nn.reset_hidden);
        result.reset_bias = autodiff::get_grad<la::vector<double>>(nn.reset_bias);

        result.update_input = autodiff::get_grad<la::matrix<double>>(nn.update_input);
        result.update_hidden = autodiff::get_grad<la::matrix<double>>(nn.update_hidden);
        result.update_bias = autodiff::get_grad<la::vector<double>>(nn.update_bias);

        result.candidate_input = autodiff::get_grad<la::matrix<double>>(nn.candidate_input);
        result.candidate_hidden = autodiff::get_grad<la::matrix<double>>(nn.candidate_hidden);
        result.candidate_bias = autodiff::get_grad<la::vector<double>>(nn.candidate_bias);

        result.shortcut_input = autodiff::get_grad<la::matrix<double>>(nn.shortcut_input);
        result.shortcut_bias = autodiff::get_grad<la::vector<double>>(nn.shortcut_bias);

        return result;
    }

    bgru_feat_param_t load_bgru_feat_param(std::istream& is)
    {
        bgru_feat_param_t result;
        std::string line;

        ebt::json::json_parser<la::matrix<double>> mat_parser;
        ebt::json::json_parser<la::vector<double>> vec_parser;

        result.forward_param = load_gru_feat_param(is);
        result.backward_param = load_gru_feat_param(is);

        result.forward_output = mat_parser.parse(is);
        std::getline(is, line);

        result.backward_output = mat_parser.parse(is);
        std::getline(is, line);

        result.output_bias = vec_parser.parse(is);
        std::getline(is, line);

        return result;
    }

    bgru_feat_param_t load_bgru_feat_param(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_bgru_feat_param(ifs);
    }

    void save_bgru_feat_param(bgru_feat_param_t const& param, std::ostream& os)
    {
        save_gru_feat_param(param.forward_param, os);
        save_gru_feat_param(param.backward_param, os);

        ebt::json::dump(param.forward_output, os);
        os << std::endl;
        ebt::json::dump(param.backward_output, os);
        os << std::endl;
        ebt::json::dump(param.output_bias, os);
        os << std::endl;
    }

    void save_bgru_feat_param(bgru_feat_param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };

        save_bgru_feat_param(param, ofs);
    }

    void adagrad_update(bgru_feat_param_t& param, bgru_feat_param_t const& grad,
        bgru_feat_param_t& opt_data, double step_size)
    {
        adagrad_update(param.forward_param, grad.forward_param,
            opt_data.forward_param, step_size);
        adagrad_update(param.backward_param, grad.backward_param,
            opt_data.backward_param, step_size);

        opt::adagrad_update(param.forward_output, grad.forward_output,
            opt_data.forward_output, step_size);
        opt::adagrad_update(param.backward_output, grad.backward_output,
            opt_data.backward_output, step_size);
        opt::adagrad_update(param.output_bias, grad.output_bias,
            opt_data.output_bias, step_size);
    }

    void rmsprop_update(bgru_feat_param_t& param, bgru_feat_param_t const& grad,
        bgru_feat_param_t& opt_data, double decay, double step_size)
    {
        rmsprop_update(param.forward_param, grad.forward_param,
            opt_data.forward_param, decay, step_size);
        rmsprop_update(param.backward_param, grad.backward_param,
            opt_data.backward_param, decay, step_size);

        opt::rmsprop_update(param.forward_output, grad.forward_output,
            opt_data.forward_output, decay, step_size);
        opt::rmsprop_update(param.backward_output, grad.backward_output,
            opt_data.backward_output, decay, step_size);
        opt::rmsprop_update(param.output_bias, grad.output_bias,
            opt_data.output_bias, decay, step_size);
    }

    bgru_feat_nn_t make_bgru_feat_nn(autodiff::computation_graph& g,
        bgru_feat_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        bgru_feat_nn_t result;

        result.forward_nn = make_gru_feat_nn(g, param.forward_param, inputs);

        std::vector<std::shared_ptr<autodiff::op_t>> rev_inputs = inputs;
        std::reverse(rev_inputs.begin(), rev_inputs.end());

        result.backward_nn = make_gru_feat_nn(g, param.forward_param, rev_inputs);

        std::reverse(result.backward_nn.hidden.begin(), result.backward_nn.hidden.end());
        std::reverse(result.backward_nn.candidate.begin(), result.backward_nn.candidate.end());
        std::reverse(result.backward_nn.update.begin(), result.backward_nn.update.end());
        std::reverse(result.backward_nn.reset.begin(), result.backward_nn.reset.end());

        result.forward_output = g.var(param.forward_output);
        result.backward_output = g.var(param.backward_output);
        result.output_bias = g.var(param.output_bias);

        for (int i = 0; i < result.forward_nn.hidden.size(); ++i) {
            result.output.push_back(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.forward_output, result.forward_nn.hidden[i]),
                    autodiff::mul(result.backward_output, result.backward_nn.hidden[i]),
                    result.output_bias
                }));
        }

        return result;
    }

    bgru_feat_param_t copy_grad(bgru_feat_nn_t const& nn)
    {
        bgru_feat_param_t result;

        result.forward_param = copy_grad(nn.forward_nn);
        result.backward_param = copy_grad(nn.backward_nn);

        result.forward_output = autodiff::get_grad<la::matrix<double>>(nn.forward_output);
        result.backward_output = autodiff::get_grad<la::matrix<double>>(nn.backward_output);
        result.output_bias = autodiff::get_grad<la::vector<double>>(nn.output_bias);

        return result;
    }

    dbgru_feat_param_t load_dbgru_feat_param(std::istream& is)
    {
        dbgru_feat_param_t result;

        std::string line;
        std::getline(is, line);

        int layers = std::stoi(line);

        for (int i = 0; i < layers; ++i) {
            result.layer.push_back(load_bgru_feat_param(is));
        }

        return result;
    }

    dbgru_feat_param_t load_dbgru_feat_param(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_dbgru_feat_param(ifs);
    }

    void save_dbgru_feat_param(dbgru_feat_param_t const& param, std::ostream& os)
    {
        os << param.layer.size() << std::endl;

        for (int i = 0; i < param.layer.size(); ++i) {
            save_bgru_feat_param(param.layer[i], os);
        }
    }

    void save_dbgru_feat_param(dbgru_feat_param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };

        save_dbgru_feat_param(param, ofs);
    }

    void adagrad_update(dbgru_feat_param_t& param, dbgru_feat_param_t const& grad,
        dbgru_feat_param_t& opt_data, double step_size)
    {
        for (int i = 0; i < param.layer.size(); ++i) {
            adagrad_update(param.layer[i], grad.layer[i],
                opt_data.layer[i], step_size);
        }
    }

    void rmsprop_update(dbgru_feat_param_t& param, dbgru_feat_param_t const& grad,
        dbgru_feat_param_t& opt_data, double decay, double step_size)
    {
        for (int i = 0; i < param.layer.size(); ++i) {
            rmsprop_update(param.layer[i], grad.layer[i],
                opt_data.layer[i], decay, step_size);
        }
    }

    dbgru_feat_nn_t make_dbgru_feat_nn(autodiff::computation_graph& g,
        dbgru_feat_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        dbgru_feat_nn_t result;

        result.layer.push_back(make_bgru_feat_nn(g, param.layer[0], inputs));
        for (int i = 1; i < param.layer.size(); ++i) {
            result.layer.push_back(make_bgru_feat_nn(g, param.layer[i], result.layer.back().output));
        }

        return result;
    }

    dbgru_feat_param_t copy_grad(dbgru_feat_nn_t const& nn)
    {
        dbgru_feat_param_t result;

        for (int i = 0; i < nn.layer.size(); ++i) {
            result.layer.push_back(copy_grad(nn.layer[i]));
        }

        return result;
    }

    pred_param_t load_pred_param(std::istream& is)
    {
        pred_param_t result;
        std::string line;

        ebt::json::json_parser<la::matrix<double>> mat_parser;
        ebt::json::json_parser<la::vector<double>> vec_parser;

        result.softmax_weight = mat_parser.parse(is);
        std::getline(is, line);
        result.softmax_bias = vec_parser.parse(is);
        std::getline(is, line);

        return result;
    }

    pred_param_t load_pred_param(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_pred_param(ifs);
    }

    void save_pred_param(pred_param_t const& param, std::ostream& os)
    {
        ebt::json::dump(param.softmax_weight, os);
        os << std::endl;
        ebt::json::dump(param.softmax_bias, os);
        os << std::endl;
    }

    void save_pred_param(pred_param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };

        save_pred_param(param, ofs);
    }

    void adagrad_update(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double step_size)
    {
        opt::adagrad_update(param.softmax_weight, grad.softmax_weight,
            opt_data.softmax_weight, step_size);
        opt::adagrad_update(param.softmax_bias, grad.softmax_bias,
            opt_data.softmax_bias, step_size);
    }

    void rmsprop_update(pred_param_t& param, pred_param_t const& grad,
        pred_param_t& opt_data, double decay, double step_size)
    {
        opt::rmsprop_update(param.softmax_weight, grad.softmax_weight,
            opt_data.softmax_weight, decay, step_size);
        opt::rmsprop_update(param.softmax_bias, grad.softmax_bias,
            opt_data.softmax_bias, decay, step_size);
    }

    pred_nn_t make_pred_nn(autodiff::computation_graph& g,
        pred_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat)
    {
        pred_nn_t result;

        result.softmax_weight = g.var(param.softmax_weight);
        result.softmax_bias = g.var(param.softmax_bias);

        for (int i = 0; i < feat.size(); ++i) {
            result.logprob.push_back(autodiff::logsoftmax(autodiff::add(
                autodiff::mul(result.softmax_weight, feat[i]), result.softmax_bias)));
        }

        return result;
    }

    pred_param_t copy_grad(pred_nn_t const& nn)
    {
        pred_param_t result;

        result.softmax_weight = autodiff::get_grad<la::matrix<double>>(nn.softmax_weight);
        result.softmax_bias = autodiff::get_grad<la::vector<double>>(nn.softmax_bias);

        return result;
    }

    void eval(pred_nn_t& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::eval(order, autodiff::eval_funcs);
    }

    void grad(pred_nn_t& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::grad(order, autodiff::grad_funcs);
    }

    double log_loss::loss()
    {
        return -la::dot(gold, pred);
    }

    la::vector<double> log_loss::grad()
    {
        return la::mul(gold, -1);
    }

}
