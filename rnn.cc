#include "nn/rnn.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace lstm {

    void bound(la::vector_like<double>& u, double min, double max)
    {
        for (int i = 0; i < u.size(); ++i) {
            u(i) = std::max(min, std::min(max, u(i)));
        }
    }

    void bound(la::matrix_like<double>& u, double min, double max)
    {
        la::weak_vector<double> v { u.data(), u.rows() * u.cols() };
        bound(v, min, max);
    }

    lstm_feat_param_t load_lstm_feat_param(std::istream& is)
    {
        std::string line;
        lstm_feat_param_t result;

        result.hidden_input = ebt::json::json_parser<decltype(result.hidden_input)>().parse(is);
        std::getline(is, line);
        result.hidden_output = ebt::json::json_parser<decltype(result.hidden_output)>().parse(is);
        std::getline(is, line);
        result.hidden_bias = ebt::json::json_parser<decltype(result.hidden_bias)>().parse(is);
        std::getline(is, line);

        result.input_input = ebt::json::json_parser<decltype(result.input_input)>().parse(is);
        std::getline(is, line);
        result.input_output = ebt::json::json_parser<decltype(result.input_output)>().parse(is);
        std::getline(is, line);
        result.input_peep = ebt::json::json_parser<decltype(result.input_peep)>().parse(is);
        std::getline(is, line);
        result.input_bias = ebt::json::json_parser<decltype(result.input_bias)>().parse(is);
        std::getline(is, line);

        result.output_input = ebt::json::json_parser<decltype(result.output_input)>().parse(is);
        std::getline(is, line);
        result.output_output = ebt::json::json_parser<decltype(result.output_output)>().parse(is);
        std::getline(is, line);
        result.output_peep = ebt::json::json_parser<decltype(result.output_peep)>().parse(is);
        std::getline(is, line);
        result.output_bias = ebt::json::json_parser<decltype(result.output_bias)>().parse(is);
        std::getline(is, line);

        result.forget_input = ebt::json::json_parser<decltype(result.forget_input)>().parse(is);
        std::getline(is, line);
        result.forget_output = ebt::json::json_parser<decltype(result.forget_output)>().parse(is);
        std::getline(is, line);
        result.forget_peep = ebt::json::json_parser<decltype(result.forget_peep)>().parse(is);
        std::getline(is, line);
        result.forget_bias = ebt::json::json_parser<decltype(result.forget_bias)>().parse(is);
        std::getline(is, line);

        return result;
    }

    lstm_feat_param_t load_lstm_feat_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_lstm_feat_param(ifs);
    }

    void save_lstm_feat_param(lstm_feat_param_t const& p, std::ostream& os)
    {
        ebt::json::dump(p.hidden_input, os);
        os << std::endl;
        ebt::json::dump(p.hidden_output, os);
        os << std::endl;
        ebt::json::dump(p.hidden_bias, os);
        os << std::endl;

        ebt::json::dump(p.input_input, os);
        os << std::endl;
        ebt::json::dump(p.input_output, os);
        os << std::endl;
        ebt::json::dump(p.input_peep, os);
        os << std::endl;
        ebt::json::dump(p.input_bias, os);
        os << std::endl;

        ebt::json::dump(p.output_input, os);
        os << std::endl;
        ebt::json::dump(p.output_output, os);
        os << std::endl;
        ebt::json::dump(p.output_peep, os);
        os << std::endl;
        ebt::json::dump(p.output_bias, os);
        os << std::endl;

        ebt::json::dump(p.forget_input, os);
        os << std::endl;
        ebt::json::dump(p.forget_output, os);
        os << std::endl;
        ebt::json::dump(p.forget_peep, os);
        os << std::endl;
        ebt::json::dump(p.forget_bias, os);
        os << std::endl;
    }

    void save_lstm_feat_param(lstm_feat_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_lstm_feat_param(p, ofs);
    }

    void bound(lstm_feat_param_t& p, double min, double max)
    {
        bound(p.hidden_input, min, max);
        bound(p.hidden_output, min, max);
        bound(p.hidden_bias, min, max);

        bound(p.input_input, min, max);
        bound(p.input_output, min, max);
        bound(p.input_peep, min, max);
        bound(p.input_bias, min, max);

        bound(p.output_input, min, max);
        bound(p.output_output, min, max);
        bound(p.output_peep, min, max);
        bound(p.output_bias, min, max);

        bound(p.forget_input, min, max);
        bound(p.forget_output, min, max);
        bound(p.forget_peep, min, max);
        bound(p.forget_bias, min, max);
    }

    void const_step_update_momentum(lstm_feat_param_t& p, lstm_feat_param_t const& grad,
        lstm_feat_param_t& opt_data, double momentum, double step_size)
    {
        opt::const_step_update_momentum(p.hidden_input, grad.hidden_input,
            opt_data.hidden_input, momentum, step_size);
        opt::const_step_update_momentum(p.hidden_output, grad.hidden_output,
            opt_data.hidden_output, momentum, step_size);
        opt::const_step_update_momentum(p.hidden_bias, grad.hidden_bias,
            opt_data.hidden_bias, momentum, step_size);

        opt::const_step_update_momentum(p.input_input, grad.input_input,
            opt_data.input_input, momentum, step_size);
        opt::const_step_update_momentum(p.input_output, grad.input_output,
            opt_data.input_output, momentum, step_size);
        opt::const_step_update_momentum(p.input_peep, grad.input_peep,
            opt_data.input_peep, momentum, step_size);
        opt::const_step_update_momentum(p.input_bias, grad.input_bias,
            opt_data.input_bias, momentum, step_size);

        opt::const_step_update_momentum(p.output_input, grad.output_input,
            opt_data.output_input, momentum, step_size);
        opt::const_step_update_momentum(p.output_output, grad.output_output,
            opt_data.output_output, momentum, step_size);
        opt::const_step_update_momentum(p.output_peep, grad.output_peep,
            opt_data.output_peep, momentum, step_size);
        opt::const_step_update_momentum(p.output_bias, grad.output_bias,
            opt_data.output_bias, momentum, step_size);

        opt::const_step_update_momentum(p.forget_input, grad.forget_input,
            opt_data.forget_input, momentum, step_size);
        opt::const_step_update_momentum(p.forget_output, grad.forget_output,
            opt_data.forget_output, momentum, step_size);
        opt::const_step_update_momentum(p.forget_peep, grad.forget_peep,
            opt_data.forget_peep, momentum, step_size);
        opt::const_step_update_momentum(p.forget_bias, grad.forget_bias,
            opt_data.forget_bias, momentum, step_size);
    }

    void adagrad_update(lstm_feat_param_t& p, lstm_feat_param_t const& grad,
        lstm_feat_param_t& opt_data, double step_size)
    {
        opt::adagrad_update(p.hidden_input, grad.hidden_input,
            opt_data.hidden_input, step_size);
        opt::adagrad_update(p.hidden_output, grad.hidden_output,
            opt_data.hidden_output, step_size);
        opt::adagrad_update(p.hidden_bias, grad.hidden_bias,
            opt_data.hidden_bias, step_size);

        opt::adagrad_update(p.input_input, grad.input_input,
            opt_data.input_input, step_size);
        opt::adagrad_update(p.input_output, grad.input_output,
            opt_data.input_output, step_size);
        opt::adagrad_update(p.input_peep, grad.input_peep,
            opt_data.input_peep, step_size);
        opt::adagrad_update(p.input_bias, grad.input_bias,
            opt_data.input_bias, step_size);

        opt::adagrad_update(p.output_input, grad.output_input,
            opt_data.output_input, step_size);
        opt::adagrad_update(p.output_output, grad.output_output,
            opt_data.output_output, step_size);
        opt::adagrad_update(p.output_peep, grad.output_peep,
            opt_data.output_peep, step_size);
        opt::adagrad_update(p.output_bias, grad.output_bias,
            opt_data.output_bias, step_size);

        opt::adagrad_update(p.forget_input, grad.forget_input,
            opt_data.forget_input, step_size);
        opt::adagrad_update(p.forget_output, grad.forget_output,
            opt_data.forget_output, step_size);
        opt::adagrad_update(p.forget_peep, grad.forget_peep,
            opt_data.forget_peep, step_size);
        opt::adagrad_update(p.forget_bias, grad.forget_bias,
            opt_data.forget_bias, step_size);
    }

    lstm_feat_nn_t make_forward_lstm_feat_nn(autodiff::computation_graph& g,
        lstm_feat_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        lstm_feat_nn_t result;

        result.hidden_input = g.var(p.hidden_input);
        result.hidden_output = g.var(p.hidden_output);
        result.hidden_bias = g.var(p.hidden_bias);

        result.input_input = g.var(p.input_input);
        result.input_output = g.var(p.input_output);
        result.input_peep = g.var(p.input_peep);
        result.input_bias = g.var(p.input_bias);

        result.output_input = g.var(p.output_input);
        result.output_output = g.var(p.output_output);
        result.output_peep = g.var(p.output_peep);
        result.output_bias = g.var(p.output_bias);

        result.forget_input = g.var(p.forget_input);
        result.forget_output = g.var(p.forget_output);
        result.forget_peep = g.var(p.forget_peep);
        result.forget_bias = g.var(p.forget_bias);

        result.hidden.push_back(autodiff::tanh(
            autodiff::add(autodiff::mul(result.hidden_input, inputs.front()),
            result.hidden_bias)));

        result.input_gate.push_back(autodiff::logistic(
            autodiff::add(autodiff::mul(result.input_input, inputs.front()),
            result.input_bias)));

        result.cell.push_back(autodiff::emul(result.input_gate.back(),
            result.hidden.back()));

        result.output_gate.push_back(autodiff::logistic(autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.output_input, inputs.front()),
                autodiff::emul(result.output_peep, result.cell.back()),
                result.output_bias
            })));

        result.output.push_back(autodiff::emul(result.output_gate.back(),
            autodiff::tanh(result.cell.back())));

        for (int i = 1; i < inputs.size(); ++i) {
            result.hidden.push_back(autodiff::tanh(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.hidden_input, inputs[i]),
                    autodiff::mul(result.hidden_output, result.output.back()),
                    result.hidden_bias
                })));

            result.input_gate.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.input_input, inputs[i]),
                    autodiff::mul(result.input_output, result.output.back()),
                    autodiff::emul(result.input_peep, result.cell.back()),
                    result.input_bias
                })));

            result.forget_gate.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.forget_input, inputs[i]),
                    autodiff::mul(result.forget_output, result.output.back()),
                    autodiff::emul(result.forget_peep, result.cell.back()),
                    result.forget_bias
                })));

            result.cell.push_back(autodiff::add(
                autodiff::emul(result.forget_gate.back(), result.cell.back()),
                autodiff::emul(result.input_gate.back(), result.hidden.back())));

            result.output_gate.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.output_input, inputs[i]),
                    autodiff::mul(result.output_output, result.output.back()),
                    autodiff::emul(result.output_peep, result.cell.back()),
                    result.output_bias
                })));

            result.output.push_back(autodiff::emul(result.output_gate.back(),
                autodiff::tanh(result.cell.back())));
        }

        return result;
    }

    lstm_feat_nn_t make_backward_lstm_feat_nn(autodiff::computation_graph& g,
        lstm_feat_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> rev_inputs = inputs;
        std::reverse(rev_inputs.begin(), rev_inputs.end());

        lstm_feat_nn_t result = make_forward_lstm_feat_nn(g, p, rev_inputs);

        std::reverse(result.cell.begin(), result.cell.end());
        std::reverse(result.hidden.begin(), result.hidden.end());
        std::reverse(result.input_gate.begin(), result.input_gate.end());
        std::reverse(result.output_gate.begin(), result.output_gate.end());
        std::reverse(result.forget_gate.begin(), result.forget_gate.end());
        std::reverse(result.output.begin(), result.output.end());

        return result;
    }

    lstm_feat_param_t copy_lstm_feat_grad(lstm_feat_nn_t const& nn)
    {
        lstm_feat_param_t result;

        result.hidden_input = autodiff::get_grad<decltype(result.hidden_input)>(nn.hidden_input);
        result.hidden_output = autodiff::get_grad<decltype(result.hidden_output)>(nn.hidden_output);
        result.hidden_bias = autodiff::get_grad<decltype(result.hidden_bias)>(nn.hidden_bias);

        result.input_input = autodiff::get_grad<decltype(result.input_input)>(nn.input_input);
        result.input_output = autodiff::get_grad<decltype(result.input_output)>(nn.input_output);
        result.input_peep = autodiff::get_grad<decltype(result.input_peep)>(nn.input_peep);
        result.input_bias = autodiff::get_grad<decltype(result.input_bias)>(nn.input_bias);

        result.output_input = autodiff::get_grad<decltype(result.output_input)>(nn.output_input);
        result.output_output = autodiff::get_grad<decltype(result.output_output)>(nn.output_output);
        result.output_peep = autodiff::get_grad<decltype(result.output_peep)>(nn.output_peep);
        result.output_bias = autodiff::get_grad<decltype(result.output_bias)>(nn.output_bias);

        result.forget_input = autodiff::get_grad<decltype(result.forget_input)>(nn.forget_input);
        result.forget_output = autodiff::get_grad<decltype(result.forget_output)>(nn.forget_output);
        result.forget_peep = autodiff::get_grad<decltype(result.forget_peep)>(nn.forget_peep);
        result.forget_bias = autodiff::get_grad<decltype(result.forget_bias)>(nn.forget_bias);

        return result;
    }

    lstm_param_t load_lstm_param(std::istream& is)
    {
        std::string line;
        lstm_param_t result;

        result.feat_param = load_lstm_feat_param(is);

        result.softmax_weight = ebt::json::json_parser<decltype(result.softmax_weight)>().parse(is);
        std::getline(is, line);
        result.softmax_bias = ebt::json::json_parser<decltype(result.softmax_bias)>().parse(is);
        std::getline(is, line);

        return result;
    }

    lstm_param_t load_lstm_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_lstm_param(ifs);
    }

    void save_lstm_param(lstm_param_t const& p, std::ostream& os)
    {
        save_lstm_feat_param(p.feat_param, os);

        ebt::json::dump(p.softmax_weight, os);
        os << std::endl;
        ebt::json::dump(p.softmax_bias, os);
        os << std::endl;
    }

    void save_lstm_param(lstm_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_lstm_param(p, ofs);
    }

    void adagrad_update(lstm_param_t& p, lstm_param_t const& grad,
        lstm_param_t& opt_data, double step_size)
    {
        adagrad_update(p.feat_param, grad.feat_param, opt_data.feat_param, step_size);

        opt::adagrad_update(p.softmax_weight, grad.softmax_weight,
            opt_data.softmax_weight, step_size);
        opt::adagrad_update(p.softmax_bias, grad.softmax_bias,
            opt_data.softmax_bias, step_size);
    }

    lstm_nn_t make_lstm_nn(lstm_param_t const& p,
        std::vector<std::vector<double>> const& frames)
    {
        lstm_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(result.graph.var(frames[i]));
        }

        result.feat_nn = make_forward_lstm_feat_nn(result.graph, p.feat_param, inputs);

        result.softmax_weight = result.graph.var(p.softmax_weight);
        result.softmax_bias = result.graph.var(p.softmax_bias);

        for (int i = 0; i < result.feat_nn.output.size(); ++i) {
            result.logprob.push_back(autodiff::logsoftmax(autodiff::add(
                autodiff::mul(result.softmax_weight, result.feat_nn.output[i]),
                result.softmax_bias)));
        }

        return result;
    }

    lstm_param_t copy_lstm_grad(lstm_nn_t const& nn)
    {
        lstm_param_t result;

        result.feat_param = copy_lstm_feat_grad(nn.feat_nn);

        result.softmax_weight = autodiff::get_grad<decltype(result.softmax_weight)>(nn.softmax_weight);
        result.softmax_bias = autodiff::get_grad<decltype(result.softmax_bias)>(nn.softmax_bias);

        return result;
    }

    void eval(lstm_nn_t const& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::eval(order, autodiff::eval_funcs);
    }

    void grad(lstm_nn_t const& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::grad(order, autodiff::grad_funcs);
    }

    blstm_feat_param_t load_blstm_feat_param(std::istream& is)
    {
        std::string line;
        blstm_feat_param_t result;

        result.forward_param = load_lstm_feat_param(is);
        result.backward_param = load_lstm_feat_param(is);

        result.forward_output_weight = ebt::json::json_parser<decltype(result.forward_output_weight)>().parse(is);
        std::getline(is, line);
        result.backward_output_weight = ebt::json::json_parser<decltype(result.backward_output_weight)>().parse(is);
        std::getline(is, line);
        result.output_bias = ebt::json::json_parser<decltype(result.output_bias)>().parse(is);
        std::getline(is, line);

        return result;
    }

    blstm_feat_param_t load_blstm_feat_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_blstm_feat_param(ifs);
    }

    void save_blstm_feat_param(blstm_feat_param_t const& p, std::ostream& os)
    {
        save_lstm_feat_param(p.forward_param, os);
        save_lstm_feat_param(p.backward_param, os);

        ebt::json::dump(p.forward_output_weight, os);
        os << std::endl;
        ebt::json::dump(p.backward_output_weight, os);
        os << std::endl;
        ebt::json::dump(p.output_bias, os);
        os << std::endl;
    }

    void save_blstm_feat_param(blstm_feat_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_blstm_feat_param(p, ofs);
    }

    void bound(blstm_feat_param_t& p, double min, double max)
    {
        bound(p.forward_param, min, max);
        bound(p.backward_param, min, max);

        bound(p.forward_output_weight, min, max);
        bound(p.backward_output_weight, min, max);
        bound(p.output_bias, min, max);
    }

    void const_step_update_momentum(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
        blstm_feat_param_t& opt_data, double momentum, double step_size)
    {
        const_step_update_momentum(p.forward_param, grad.forward_param,
            opt_data.forward_param, momentum, step_size);
        const_step_update_momentum(p.backward_param, grad.backward_param,
            opt_data.backward_param, momentum, step_size);

        opt::const_step_update_momentum(p.forward_output_weight, grad.forward_output_weight,
            opt_data.forward_output_weight, momentum, step_size);
        opt::const_step_update_momentum(p.backward_output_weight, grad.backward_output_weight,
            opt_data.backward_output_weight, momentum, step_size);
        opt::const_step_update_momentum(p.output_bias, grad.output_bias,
            opt_data.output_bias, momentum, step_size);
    }

    void adagrad_update(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
        blstm_feat_param_t& opt_data, double step_size)
    {
        adagrad_update(p.forward_param, grad.forward_param,
            opt_data.forward_param, step_size);
        adagrad_update(p.backward_param, grad.backward_param,
            opt_data.backward_param, step_size);

        opt::adagrad_update(p.forward_output_weight, grad.forward_output_weight,
            opt_data.forward_output_weight, step_size);
        opt::adagrad_update(p.backward_output_weight, grad.backward_output_weight,
            opt_data.backward_output_weight, step_size);
        opt::adagrad_update(p.output_bias, grad.output_bias,
            opt_data.output_bias, step_size);
    }

    blstm_param_t load_blstm_param(std::istream& is)
    {
        std::string line;
        blstm_param_t result;

        result.feat_param = load_blstm_feat_param(is);

        result.softmax_weight = ebt::json::json_parser<decltype(result.softmax_weight)>().parse(is);
        std::getline(is, line);
        result.softmax_bias = ebt::json::json_parser<decltype(result.softmax_bias)>().parse(is);
        std::getline(is, line);

        return result;
    }

    blstm_param_t load_blstm_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_blstm_param(ifs);
    }

    void save_blstm_param(blstm_param_t const& p, std::ostream& os)
    {
        save_blstm_feat_param(p.feat_param, os);

        ebt::json::dump(p.softmax_weight, os);
        os << std::endl;
        ebt::json::dump(p.softmax_bias, os);
        os << std::endl;
    }

    void save_blstm_param(blstm_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_blstm_param(p, ofs);
    }

    void adagrad_update(blstm_param_t& p, blstm_param_t const& grad,
        blstm_param_t& opt_data, double step_size)
    {
        adagrad_update(p.feat_param, grad.feat_param,
            opt_data.feat_param, step_size);

        opt::adagrad_update(p.softmax_weight, grad.softmax_weight,
            opt_data.softmax_weight, step_size);
        opt::adagrad_update(p.softmax_bias, grad.softmax_bias,
            opt_data.softmax_bias, step_size);
    }

    blstm_feat_nn_t make_blstm_feat_nn(autodiff::computation_graph& g,
        blstm_feat_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        blstm_feat_nn_t result;

        result.forward_feat_nn = make_forward_lstm_feat_nn(g, p.forward_param, inputs);
        result.backward_feat_nn = make_backward_lstm_feat_nn(g, p.backward_param, inputs);

        result.forward_output_weight = g.var(p.forward_output_weight);
        result.backward_output_weight = g.var(p.backward_output_weight);
        result.output_bias = g.var(p.output_bias);

        for (int i = 0; i < result.forward_feat_nn.output.size(); ++i) {
            result.output.push_back(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.forward_output_weight, result.forward_feat_nn.output[i]),
                    autodiff::mul(result.backward_output_weight, result.backward_feat_nn.output[i]),
                    result.output_bias
                }));
        }

        return result;
    }

    blstm_feat_param_t copy_blstm_feat_grad(blstm_feat_nn_t const& nn)
    {
        blstm_feat_param_t result;

        result.forward_param = copy_lstm_feat_grad(nn.forward_feat_nn);
        result.backward_param = copy_lstm_feat_grad(nn.backward_feat_nn);

        result.forward_output_weight = autodiff::get_grad<decltype(result.forward_output_weight)>(nn.forward_output_weight);
        result.backward_output_weight = autodiff::get_grad<decltype(result.backward_output_weight)>(nn.backward_output_weight);
        result.output_bias = autodiff::get_grad<decltype(result.output_bias)>(nn.output_bias);

        return result;
    }

    blstm_nn_t make_blstm_nn(blstm_param_t const& p,
        std::vector<std::vector<double>> const& frames)
    {
        blstm_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(result.graph.var(frames[i]));
        }

        result.feat_nn = make_blstm_feat_nn(result.graph, p.feat_param, inputs);

        result.softmax_weight = result.graph.var(p.softmax_weight);
        result.softmax_bias = result.graph.var(p.softmax_bias);

        for (int i = 0; i < result.feat_nn.output.size(); ++i) {
            result.logprob.push_back(autodiff::logsoftmax(autodiff::add(
                autodiff::mul(result.softmax_weight, result.feat_nn.output[i]),
                result.softmax_bias)));
        }

        return result;
    }

    blstm_param_t copy_blstm_grad(blstm_nn_t const& nn)
    {
        blstm_param_t result;

        result.feat_param = copy_blstm_feat_grad(nn.feat_nn);

        result.softmax_weight = autodiff::get_grad<decltype(result.softmax_weight)>(nn.softmax_weight);
        result.softmax_bias = autodiff::get_grad<decltype(result.softmax_bias)>(nn.softmax_bias);

        return result;
    }

    void eval(blstm_nn_t const& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::eval(order, autodiff::eval_funcs);
    }

    void grad(blstm_nn_t const& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::grad(order, autodiff::grad_funcs);
    }

    dblstm_param_t load_dblstm_param(std::istream& is)
    {
        std::string line;
        std::getline(is, line);

        dblstm_param_t result;

        int layer = std::stoi(line);

        for (int i = 0; i < layer; ++i) {
            result.layer.push_back(load_blstm_feat_param(is));
        }

        result.softmax_weight = ebt::json::json_parser<decltype(result.softmax_weight)>().parse(is);
        std::getline(is, line);

        result.softmax_bias = ebt::json::json_parser<decltype(result.softmax_bias)>().parse(is);
        std::getline(is, line);

        return result;
    }

    dblstm_param_t load_dblstm_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_dblstm_param(ifs);
    }

    void save_dblstm_param(dblstm_param_t const& p, std::ostream& os)
    {
        os << p.layer.size() << std::endl;

        for (auto& ell: p.layer) {
            save_blstm_feat_param(ell, os);
        }

        ebt::json::dump(p.softmax_weight, os);
        os << std::endl;
        ebt::json::dump(p.softmax_bias, os);
        os << std::endl;
    }

    void save_dblstm_param(dblstm_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_dblstm_param(p, ofs);
    }

    void bound(dblstm_param_t& p, double min, double max)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            bound(p.layer[i], min, max);
        }

        bound(p.softmax_weight, min, max);
        bound(p.softmax_bias, min, max);
    }

    void const_step_update_momentum(dblstm_param_t& p, dblstm_param_t const& grad,
        dblstm_param_t& opt_data, double momentum, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            const_step_update_momentum(p.layer[i], grad.layer[i],
                opt_data.layer[i], momentum, step_size);
        }

        opt::const_step_update_momentum(p.softmax_weight, grad.softmax_weight,
            opt_data.softmax_weight, momentum, step_size);
        opt::const_step_update_momentum(p.softmax_bias, grad.softmax_bias,
            opt_data.softmax_bias, momentum, step_size);
    }

    void adagrad_update(dblstm_param_t& p, dblstm_param_t const& grad,
        dblstm_param_t& opt_data, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            adagrad_update(p.layer[i], grad.layer[i],
                opt_data.layer[i], step_size);
        }

        opt::adagrad_update(p.softmax_weight, grad.softmax_weight,
            opt_data.softmax_weight, step_size);
        opt::adagrad_update(p.softmax_bias, grad.softmax_bias,
            opt_data.softmax_bias, step_size);
    }

    dblstm_nn_t make_dblstm_nn(dblstm_param_t const& p,
        std::vector<std::vector<double>> const& frames)
    {
        dblstm_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(result.graph.var(la::vector<double>(frames[i])));
        }

        for (int i = 0; i < p.layer.size(); ++i) {
            if (i == 0) {
                result.layer.push_back(make_blstm_feat_nn(result.graph, p.layer[0], inputs));
            } else {
                result.layer.push_back(make_blstm_feat_nn(result.graph, p.layer[i],
                    result.layer[i-1].output));
            }
        }

        result.softmax_weight = result.graph.var(p.softmax_weight);
        result.softmax_bias = result.graph.var(p.softmax_bias);

        for (int i = 0; i < result.layer.back().output.size(); ++i) {
            result.logprob.push_back(autodiff::logsoftmax(autodiff::add(
                autodiff::mul(result.softmax_weight, result.layer.back().output[i]),
                result.softmax_bias)));
        }

        return result;
    }

    dblstm_param_t copy_dblstm_grad(dblstm_nn_t const& nn)
    {
        dblstm_param_t result;

        for (int i = 0; i < nn.layer.size(); ++i) {
            result.layer.push_back(copy_blstm_feat_grad(nn.layer[i]));
        }

        result.softmax_weight = autodiff::get_grad<la::matrix<double>>(nn.softmax_weight);
        result.softmax_bias = autodiff::get_grad<la::vector<double>>(nn.softmax_bias);

        return result;
    }

    void eval(dblstm_nn_t const& nn)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> order
            = autodiff::topo_order(nn.logprob);

        autodiff::eval(order, autodiff::eval_funcs);
    }

    void grad(dblstm_nn_t const& nn)
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
