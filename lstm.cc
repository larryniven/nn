#include "nn/lstm.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace lstm {

    lstm_unit_param_t load_lstm_unit_param(std::istream& is)
    {
        std::string line;
        lstm_unit_param_t result;

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

    lstm_unit_param_t load_lstm_unit_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_lstm_unit_param(ifs);
    }

    void save_lstm_unit_param(lstm_unit_param_t const& p, std::ostream& os)
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

    void save_lstm_unit_param(lstm_unit_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_lstm_unit_param(p, ofs);
    }

    void const_step_update_momentum(lstm_unit_param_t& p, lstm_unit_param_t const& grad,
        lstm_unit_param_t& opt_data, double momentum, double step_size)
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

    void adagrad_update(lstm_unit_param_t& p, lstm_unit_param_t const& grad,
        lstm_unit_param_t& opt_data, double step_size)
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

    void rmsprop_update(lstm_unit_param_t& p, lstm_unit_param_t const& grad,
        lstm_unit_param_t& opt_data, double decay, double step_size)
    {
        opt::rmsprop_update(p.hidden_input, grad.hidden_input,
            opt_data.hidden_input, decay, step_size);
        opt::rmsprop_update(p.hidden_output, grad.hidden_output,
            opt_data.hidden_output, decay, step_size);
        opt::rmsprop_update(p.hidden_bias, grad.hidden_bias,
            opt_data.hidden_bias, decay, step_size);

        opt::rmsprop_update(p.input_input, grad.input_input,
            opt_data.input_input, decay, step_size);
        opt::rmsprop_update(p.input_output, grad.input_output,
            opt_data.input_output, decay, step_size);
        opt::rmsprop_update(p.input_peep, grad.input_peep,
            opt_data.input_peep, decay, step_size);
        opt::rmsprop_update(p.input_bias, grad.input_bias,
            opt_data.input_bias, decay, step_size);

        opt::rmsprop_update(p.output_input, grad.output_input,
            opt_data.output_input, decay, step_size);
        opt::rmsprop_update(p.output_output, grad.output_output,
            opt_data.output_output, decay, step_size);
        opt::rmsprop_update(p.output_peep, grad.output_peep,
            opt_data.output_peep, decay, step_size);
        opt::rmsprop_update(p.output_bias, grad.output_bias,
            opt_data.output_bias, decay, step_size);

        opt::rmsprop_update(p.forget_input, grad.forget_input,
            opt_data.forget_input, decay, step_size);
        opt::rmsprop_update(p.forget_output, grad.forget_output,
            opt_data.forget_output, decay, step_size);
        opt::rmsprop_update(p.forget_peep, grad.forget_peep,
            opt_data.forget_peep, decay, step_size);
        opt::rmsprop_update(p.forget_bias, grad.forget_bias,
            opt_data.forget_bias, decay, step_size);
    }

    lstm_unit_nn_t make_lstm_unit_nn(autodiff::computation_graph& g,
        lstm_unit_param_t const& p)
    {
        lstm_unit_nn_t result;

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

        return result;
    }

    lstm_unit_param_t copy_lstm_unit_grad(lstm_unit_nn_t const& nn)
    {
        lstm_unit_param_t result;

        if (nn.hidden_input->grad != nullptr) {
            result.hidden_input = autodiff::get_grad<decltype(result.hidden_input)>(nn.hidden_input);
        } else {
            auto& m = autodiff::get_output<decltype(result.hidden_input)>(nn.hidden_input);
            result.hidden_input.resize(m.rows(), m.cols());
        }

        if (nn.hidden_output->grad != nullptr) {
            result.hidden_output = autodiff::get_grad<decltype(result.hidden_output)>(nn.hidden_output);
        } else {
            auto& m = autodiff::get_output<decltype(result.hidden_output)>(nn.hidden_output);
            result.hidden_output.resize(m.rows(), m.cols());
        }

        result.hidden_bias = autodiff::get_grad<decltype(result.hidden_bias)>(nn.hidden_bias);

        if (nn.input_input->grad != nullptr) {
            result.input_input = autodiff::get_grad<decltype(result.input_input)>(nn.input_input);
        } else {
            auto& m = autodiff::get_output<decltype(result.input_input)>(nn.input_input);
            result.input_input.resize(m.rows(), m.cols());
        }

        if (nn.input_output->grad != nullptr) {
            result.input_output = autodiff::get_grad<decltype(result.input_output)>(nn.input_output);
        } else {
            auto& m = autodiff::get_output<decltype(result.input_output)>(nn.input_output);
            result.input_output.resize(m.rows(), m.cols());
        }

        if (nn.input_peep->grad != nullptr) {
            result.input_peep = autodiff::get_grad<decltype(result.input_peep)>(nn.input_peep);
        } else {
            auto& v = autodiff::get_output<decltype(result.input_peep)>(nn.input_peep);
            result.input_peep.resize(v.size());
        }

        result.input_bias = autodiff::get_grad<decltype(result.input_bias)>(nn.input_bias);

        if (nn.output_input->grad != nullptr) {
            result.output_input = autodiff::get_grad<decltype(result.output_input)>(nn.output_input);
        } else {
            auto& m = autodiff::get_output<decltype(result.output_input)>(nn.output_input);
            result.output_input.resize(m.rows(), m.cols());
        }

        if (nn.output_output->grad != nullptr) {
            result.output_output = autodiff::get_grad<decltype(result.output_output)>(nn.output_output);
        } else {
            auto& m = autodiff::get_output<decltype(result.output_output)>(nn.output_output);
            result.output_output.resize(m.rows(), m.cols());
        }

        if (nn.output_peep->grad != nullptr) {
            result.output_peep = autodiff::get_grad<decltype(result.output_peep)>(nn.output_peep);
        } else {
            auto& v = autodiff::get_output<decltype(result.output_peep)>(nn.output_peep);
            result.output_peep.resize(v.size());
        }

        result.output_bias = autodiff::get_grad<decltype(result.output_bias)>(nn.output_bias);

        if (nn.forget_input->grad != nullptr) {
            result.forget_input = autodiff::get_grad<decltype(result.forget_input)>(nn.forget_input);
        } else {
            auto& m = autodiff::get_output<decltype(result.forget_input)>(nn.forget_input);
            result.forget_input.resize(m.rows(), m.cols());
        }

        if (nn.forget_output->grad != nullptr) {
            result.forget_output = autodiff::get_grad<decltype(result.forget_output)>(nn.forget_output);
        } else {
            auto& m = autodiff::get_output<decltype(result.forget_output)>(nn.forget_output);
            result.forget_output.resize(m.rows(), m.cols());
        }

        if (nn.forget_peep->grad != nullptr) {
            result.forget_peep = autodiff::get_grad<decltype(result.forget_peep)>(nn.forget_peep);
        } else {
            auto& v = autodiff::get_output<decltype(result.forget_peep)>(nn.forget_peep);
            result.forget_peep.resize(v.size());
        }

        if (nn.forget_bias->grad != nullptr) {
            result.forget_bias = autodiff::get_grad<decltype(result.forget_bias)>(nn.forget_bias);
        } else {
            auto& v = autodiff::get_output<decltype(result.forget_bias)>(nn.forget_bias);
            result.forget_bias.resize(v.size());
        }

        return result;
    }

    lstm_step_nn_t make_lstm_step(lstm_unit_nn_t const& unit_nn,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input)
    {
        lstm_step_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> h_comp { unit_nn.hidden_bias };
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp { unit_nn.input_bias };
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate_comp { unit_nn.forget_bias };

        if (input != nullptr) {
            h_comp.push_back(autodiff::mul(unit_nn.hidden_input, input));
            input_gate_comp.push_back(autodiff::mul(unit_nn.input_input, input));
            forget_gate_comp.push_back(autodiff::mul(unit_nn.forget_input, input));
        }

        if (output != nullptr) {
            h_comp.push_back(autodiff::mul(unit_nn.hidden_output, output));
            input_gate_comp.push_back(autodiff::mul(unit_nn.input_output, output));
            forget_gate_comp.push_back(autodiff::mul(unit_nn.forget_output, output));
        }

        if (cell != nullptr) {
            input_gate_comp.push_back(autodiff::emul(unit_nn.input_peep, cell));
            forget_gate_comp.push_back(autodiff::emul(unit_nn.forget_peep, cell));
        }

        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));
        result.input_gate = autodiff::logistic(autodiff::add(input_gate_comp));
        result.forget_gate = autodiff::logistic(autodiff::add(forget_gate_comp));

        if (cell != nullptr) {
            result.cell = autodiff::add(
                autodiff::emul(result.forget_gate, cell),
                autodiff::emul(result.input_gate, h));
        } else {
            result.cell = autodiff::emul(result.input_gate, h);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_comp {
            unit_nn.output_bias, autodiff::emul(unit_nn.output_peep, result.cell) };

        if (input != nullptr) {
            output_gate_comp.push_back(autodiff::mul(unit_nn.output_input, input));
        }

        if (output != nullptr) {
            output_gate_comp.push_back(autodiff::mul(unit_nn.output_output, output));
        }

        result.output_gate = autodiff::logistic(autodiff::add(output_gate_comp));

        result.output = autodiff::emul(result.output_gate,
            autodiff::tanh(result.cell));

        return result;
    }

    lstm2d_param_t load_lstm2d_param(std::istream& is)
    {
        lstm2d_param_t result;
        std::string line;

        result.h_param = load_lstm_unit_param(is);
        result.v_param = load_lstm_unit_param(is);

        result.output_h_weight = ebt::json::json_parser<la::matrix<double>>().parse(is);
        std::getline(is, line);
        result.output_v_weight = ebt::json::json_parser<la::matrix<double>>().parse(is);
        std::getline(is, line);
        result.output_bias = ebt::json::json_parser<la::vector<double>>().parse(is);
        std::getline(is, line);

        return result;
    }

    lstm2d_param_t load_lstm2d_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_lstm2d_param(ifs);
    }

    void save_lstm2d_param(lstm2d_param_t const& p, std::ostream& os)
    {
        save_lstm_unit_param(p.h_param, os);
        save_lstm_unit_param(p.v_param, os);

        ebt::json::dump(p.output_h_weight, os);
        os << std::endl;
        ebt::json::dump(p.output_v_weight, os);
        os << std::endl;
        ebt::json::dump(p.output_bias, os);
        os << std::endl;
    }
 
    void save_lstm2d_param(lstm2d_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        return save_lstm2d_param(p, ofs);
    }

    void const_step_update_momentum(lstm2d_param_t& p, lstm2d_param_t const& grad,
        lstm2d_param_t& opt_data, double momentum, double step_size)
    {
        const_step_update_momentum(p.h_param, grad.h_param,
            opt_data.h_param, momentum, step_size);
        const_step_update_momentum(p.v_param, grad.v_param,
            opt_data.v_param, momentum, step_size);
        opt::const_step_update_momentum(p.output_h_weight, grad.output_h_weight,
            opt_data.output_h_weight, momentum, step_size);
        opt::const_step_update_momentum(p.output_v_weight, grad.output_v_weight,
            opt_data.output_v_weight, momentum, step_size);
        opt::const_step_update_momentum(p.output_bias, grad.output_bias,
            opt_data.output_bias, momentum, step_size);
    }

    void adagrad_update(lstm2d_param_t& p, lstm2d_param_t const& grad,
        lstm2d_param_t& opt_data, double step_size)
    {
        adagrad_update(p.h_param, grad.h_param, opt_data.h_param, step_size);
        adagrad_update(p.v_param, grad.v_param, opt_data.v_param, step_size);
        opt::adagrad_update(p.output_h_weight, grad.output_h_weight, opt_data.output_h_weight, step_size);
        opt::adagrad_update(p.output_v_weight, grad.output_v_weight, opt_data.output_v_weight, step_size);
        opt::adagrad_update(p.output_bias, grad.output_bias, opt_data.output_bias, step_size);
    }

    void rmsprop_update(lstm2d_param_t& p, lstm2d_param_t const& grad,
        lstm2d_param_t& opt_data, double decay, double step_size)
    {
        rmsprop_update(p.h_param, grad.h_param, opt_data.h_param, decay, step_size);
        rmsprop_update(p.v_param, grad.v_param, opt_data.v_param, decay, step_size);
        opt::rmsprop_update(p.output_h_weight, grad.output_h_weight, opt_data.output_h_weight, decay, step_size);
        opt::rmsprop_update(p.output_v_weight, grad.output_v_weight, opt_data.output_v_weight, decay, step_size);
        opt::rmsprop_update(p.output_bias, grad.output_bias, opt_data.output_bias, decay, step_size);
    }

    lstm2d_nn_t make_lstm2d_nn(autodiff::computation_graph& graph,
        lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        lstm2d_nn_t result;

        result.h_nn = make_lstm_unit_nn(graph, param.h_param);
        result.v_nn = make_lstm_unit_nn(graph, param.v_param);

        result.output_h_weight = graph.var(param.output_h_weight);
        result.output_v_weight = graph.var(param.output_v_weight);
        result.output_bias = graph.var(param.output_bias);

        std::shared_ptr<autodiff::op_t> h_cell = nullptr;
        std::shared_ptr<autodiff::op_t> h_output = nullptr;

        for (int i = 0; i < inputs.size(); ++i) {
            lstm_step_nn_t h_step = make_lstm_step(result.h_nn, h_cell, h_output, inputs[i]);
            h_cell = h_step.cell;
            h_output = h_step.output;

            result.h_cell.push_back(h_step.cell);
            result.h_output.push_back(h_step.output);

            lstm_step_nn_t v_step = make_lstm_step(result.v_nn, inputs[i], inputs[i], h_output);

            result.v_cell.push_back(v_step.cell);
            result.v_output.push_back(v_step.output);
        }

        for (int i = 0; i < result.h_cell.size(); ++i) {
            result.output.push_back(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.output_h_weight, result.h_output[i]),
                autodiff::mul(result.output_v_weight, result.v_output[i]),
                result.output_bias}));
        }

        return result;
    }

    lstm2d_param_t copy_lstm2d_grad(lstm2d_nn_t const& nn)
    {
        lstm2d_param_t result;

        result.h_param = copy_lstm_unit_grad(nn.h_nn);
        result.v_param = copy_lstm_unit_grad(nn.v_nn);

        result.output_h_weight = autodiff::get_grad<la::matrix<double>>(nn.output_h_weight);
        result.output_v_weight = autodiff::get_grad<la::matrix<double>>(nn.output_v_weight);
        result.output_bias = autodiff::get_grad<la::vector<double>>(nn.output_bias);

        return result;
    }

    // bidirectional 2-lstm

    bi_lstm2d_param_t load_bi_lstm2d_param(std::istream& is)
    {
        std::string line;
        bi_lstm2d_param_t result;

        result.forward_param = load_lstm2d_param(is);
        result.backward_param = load_lstm2d_param(is);

        return result;
    }

    bi_lstm2d_param_t load_bi_lstm2d_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_bi_lstm2d_param(ifs);
    }

    void save_bi_lstm2d_param(bi_lstm2d_param_t const& p, std::ostream& os)
    {
        save_lstm2d_param(p.forward_param, os);
        save_lstm2d_param(p.backward_param, os);
    }

    void save_bi_lstm2d_param(bi_lstm2d_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_bi_lstm2d_param(p, ofs);
    }

    void const_step_update_momentum(bi_lstm2d_param_t& p, bi_lstm2d_param_t const& grad,
        bi_lstm2d_param_t& opt_data, double momentum, double step_size)
    {
        const_step_update_momentum(p.forward_param, grad.forward_param,
            opt_data.forward_param, momentum, step_size);
        const_step_update_momentum(p.backward_param, grad.backward_param,
            opt_data.backward_param, momentum, step_size);
    }

    void adagrad_update(bi_lstm2d_param_t& p, bi_lstm2d_param_t const& grad,
        bi_lstm2d_param_t& opt_data, double step_size)
    {
        adagrad_update(p.forward_param, grad.forward_param,
            opt_data.forward_param, step_size);
        adagrad_update(p.backward_param, grad.backward_param,
            opt_data.backward_param, step_size);
    }

    void rmsprop_update(bi_lstm2d_param_t& p, bi_lstm2d_param_t const& grad,
        bi_lstm2d_param_t& opt_data, double decay, double step_size)
    {
        rmsprop_update(p.forward_param, grad.forward_param,
            opt_data.forward_param, decay, step_size);
        rmsprop_update(p.backward_param, grad.backward_param,
            opt_data.backward_param, decay, step_size);
    }

    bi_lstm2d_nn_t make_bi_lstm2d_nn(autodiff::computation_graph& graph,
        bi_lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> inputs)
    {
        bi_lstm2d_nn_t result;

        result.forward_nn = make_lstm2d_nn(graph, param.forward_param, inputs);

        std::reverse(inputs.begin(), inputs.end());

        result.backward_nn = make_lstm2d_nn(graph, param.backward_param, inputs);

        std::reverse(result.backward_nn.h_cell.begin(), result.backward_nn.h_cell.end());
        std::reverse(result.backward_nn.h_output.begin(), result.backward_nn.h_output.end());
        std::reverse(result.backward_nn.v_cell.begin(), result.backward_nn.v_cell.end());
        std::reverse(result.backward_nn.v_output.begin(), result.backward_nn.v_output.end());
        std::reverse(result.backward_nn.output.begin(), result.backward_nn.output.end());

        for (int i = 0; i < result.forward_nn.output.size(); ++i) {
            result.output.push_back(autodiff::add(result.forward_nn.output[i],
                result.backward_nn.output[i]));
        }

        return result;
    }

    bi_lstm2d_param_t copy_bi_lstm2d_grad(bi_lstm2d_nn_t const& nn)
    {
        bi_lstm2d_param_t result;

        result.forward_param = copy_lstm2d_grad(nn.forward_nn);
        result.backward_param = copy_lstm2d_grad(nn.backward_nn);

        return result;
    }

    // deep bidirectional 2-lstm

    db_lstm2d_param_t load_db_lstm2d_param(std::istream& is)
    {
        db_lstm2d_param_t result;

        std::string line;

        std::getline(is, line);
        int layer = std::stoi(line);

        for (int i = 0; i < layer; ++i) {
            result.layer.push_back(load_bi_lstm2d_param(is));
        }

        return result;
    }

    db_lstm2d_param_t load_db_lstm2d_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_db_lstm2d_param(ifs);
    }

    void save_db_lstm2d_param(db_lstm2d_param_t const& p, std::ostream& os)
    {
        os << p.layer.size() << std::endl;

        for (int i = 0; i < p.layer.size(); ++i) {
            save_bi_lstm2d_param(p.layer[i], os);
        }
    }

    void save_db_lstm2d_param(db_lstm2d_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_db_lstm2d_param(p, ofs);
    }

    void const_step_update_momentum(db_lstm2d_param_t& p, db_lstm2d_param_t const& grad,
        db_lstm2d_param_t& opt_data, double momentum, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            const_step_update_momentum(p.layer[i], grad.layer[i],
                opt_data.layer[i], momentum, step_size);
        }
    }

    void adagrad_update(db_lstm2d_param_t& p, db_lstm2d_param_t const& grad,
        db_lstm2d_param_t& opt_data, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            adagrad_update(p.layer[i], grad.layer[i], opt_data.layer[i], step_size);
        }
    }

    void rmsprop_update(db_lstm2d_param_t& p, db_lstm2d_param_t const& grad,
        db_lstm2d_param_t& opt_data, double decay, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            rmsprop_update(p.layer[i], grad.layer[i], opt_data.layer[i], decay, step_size);
        }
    }

    lstm2d_nn_t stack_lstm2d(autodiff::computation_graph& graph,
        lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs,
        std::vector<std::shared_ptr<autodiff::op_t>> const& v_output,
        std::vector<std::shared_ptr<autodiff::op_t>> const& v_cell)
    {
        lstm2d_nn_t result;

        result.h_nn = make_lstm_unit_nn(graph, param.h_param);
        result.v_nn = make_lstm_unit_nn(graph, param.v_param);

        result.output_h_weight = graph.var(param.output_h_weight);
        result.output_v_weight = graph.var(param.output_v_weight);
        result.output_bias = graph.var(param.output_bias);

        std::shared_ptr<autodiff::op_t> h_cell = nullptr;
        std::shared_ptr<autodiff::op_t> h_output = nullptr;

        for (int i = 0; i < inputs.size(); ++i) {
            lstm_step_nn_t h_step = make_lstm_step(result.h_nn, h_cell, h_output, inputs[i]);
            h_cell = h_step.cell;
            h_output = h_step.output;

            result.h_cell.push_back(h_step.cell);
            result.h_output.push_back(h_step.output);

            lstm_step_nn_t v_step = make_lstm_step(result.v_nn, v_cell[i], v_output[i], h_output);

            result.v_cell.push_back(v_step.cell);
            result.v_output.push_back(v_step.output);
        }

        for (int i = 0; i < result.h_cell.size(); ++i) {
            result.output.push_back(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.output_h_weight, result.h_output[i]),
                autodiff::mul(result.output_v_weight, result.v_output[i]),
                result.output_bias}));
        }

        return result;
    }

    bi_lstm2d_nn_t stack_bi_lstm2d(autodiff::computation_graph& graph,
        bi_lstm2d_param_t const& param, bi_lstm2d_nn_t const& prev)
    {
        bi_lstm2d_nn_t result;

        result.forward_nn = stack_lstm2d(graph, param.forward_param, prev.output,
            prev.forward_nn.v_output, prev.forward_nn.v_cell);

        std::vector<std::shared_ptr<autodiff::op_t>> rev_output = prev.output;
        std::reverse(rev_output.begin(), rev_output.end());

        result.backward_nn = stack_lstm2d(graph, param.backward_param, rev_output,
            prev.backward_nn.v_output, prev.backward_nn.v_cell);

        std::reverse(result.backward_nn.h_cell.begin(), result.backward_nn.h_cell.end());
        std::reverse(result.backward_nn.h_output.begin(), result.backward_nn.h_output.end());
        std::reverse(result.backward_nn.v_cell.begin(), result.backward_nn.v_cell.end());
        std::reverse(result.backward_nn.v_output.begin(), result.backward_nn.v_output.end());
        std::reverse(result.backward_nn.output.begin(), result.backward_nn.output.end());

        for (int i = 0; i < result.forward_nn.output.size(); ++i) {
            result.output.push_back(autodiff::add(
                result.forward_nn.output[i], result.backward_nn.output[i]));
        }

        return result;
    }

    db_lstm2d_nn_t make_db_lstm2d_nn(autodiff::computation_graph& graph,
        db_lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        db_lstm2d_nn_t result;

        result.layer.push_back(make_bi_lstm2d_nn(graph, param.layer[0], inputs));

        for (int i = 1; i < param.layer.size(); ++i) {
            result.layer.push_back(stack_bi_lstm2d(graph, param.layer[i], result.layer.back()));
        }

        return result;
    }

    db_lstm2d_param_t copy_db_lstm2d_grad(db_lstm2d_nn_t const& nn)
    {
        db_lstm2d_param_t result;

        for (int i = 0; i < nn.layer.size(); ++i) {
            result.layer.push_back(copy_bi_lstm2d_grad(nn.layer[i]));
        }

        return result;
    }

    // lstm

    lstm_feat_nn_t make_forward_lstm_feat_nn(autodiff::computation_graph& g,
        lstm_unit_param_t const& p,
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

        la::vector<double> v;
        v.resize(p.hidden_input.cols(), 1);
        result.input_mask = g.var(v);

        auto input_masked = autodiff::emul(inputs.front(), result.input_mask);

        result.hidden.push_back(autodiff::tanh(
            autodiff::add(autodiff::mul(result.hidden_input, input_masked),
                result.hidden_bias))
        );

        result.input_gate.push_back(autodiff::logistic(
            autodiff::add(autodiff::mul(result.input_input, input_masked),
                result.input_bias)));

        result.cell.push_back(autodiff::emul(result.input_gate.back(),
            result.hidden.back()));

        result.output_gate.push_back(autodiff::logistic(autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.output_input, input_masked),
                autodiff::emul(result.output_peep, result.cell.back()),
                result.output_bias
            })));

        result.output.push_back(autodiff::emul(result.output_gate.back(),
            autodiff::tanh(result.cell.back())));

        for (int i = 1; i < inputs.size(); ++i) {
            auto input_masked = autodiff::emul(inputs[i], result.input_mask);

            result.hidden.push_back(
                autodiff::tanh(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.hidden_input, input_masked),
                    autodiff::mul(result.hidden_output, result.output.back()),
                    result.hidden_bias
                }))
            );

            result.input_gate.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.input_input, input_masked),
                    autodiff::mul(result.input_output, result.output.back()),
                    autodiff::emul(result.input_peep, result.cell.back()),
                    result.input_bias
                })));

            result.forget_gate.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.forget_input, input_masked),
                    autodiff::mul(result.forget_output, result.output.back()),
                    autodiff::emul(result.forget_peep, result.cell.back()),
                    result.forget_bias
                })));

            result.cell.push_back(
                autodiff::add(
                    autodiff::emul(result.forget_gate.back(), result.cell.back()),
                    autodiff::emul(result.input_gate.back(), result.hidden.back()))
            );

            result.output_gate.push_back(autodiff::logistic(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::mul(result.output_input, input_masked),
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
        lstm_unit_param_t const& p,
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

    lstm_unit_param_t copy_lstm_feat_grad(lstm_feat_nn_t const& nn)
    {
        lstm_unit_param_t result;

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

    blstm_feat_param_t load_blstm_feat_param(std::istream& is)
    {
        std::string line;
        blstm_feat_param_t result;

        result.forward_param = load_lstm_unit_param(is);
        result.backward_param = load_lstm_unit_param(is);

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
        save_lstm_unit_param(p.forward_param, os);
        save_lstm_unit_param(p.backward_param, os);

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

    void rmsprop_update(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
        blstm_feat_param_t& opt_data, double decay, double step_size)
    {
        rmsprop_update(p.forward_param, grad.forward_param,
            opt_data.forward_param, decay, step_size);
        rmsprop_update(p.backward_param, grad.backward_param,
            opt_data.backward_param, decay, step_size);

        opt::rmsprop_update(p.forward_output_weight, grad.forward_output_weight,
            opt_data.forward_output_weight, decay, step_size);
        opt::rmsprop_update(p.backward_output_weight, grad.backward_output_weight,
            opt_data.backward_output_weight, decay, step_size);
        opt::rmsprop_update(p.output_bias, grad.output_bias,
            opt_data.output_bias, decay, step_size);
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

    dblstm_feat_param_t load_dblstm_feat_param(std::istream& is)
    {
        std::string line;
        std::getline(is, line);

        dblstm_feat_param_t result;

        int layer = std::stoi(line);

        for (int i = 0; i < layer; ++i) {
            result.layer.push_back(load_blstm_feat_param(is));
        }

        return result;
    }

    dblstm_feat_param_t load_dblstm_feat_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_dblstm_feat_param(ifs);
    }

    void save_dblstm_feat_param(dblstm_feat_param_t const& p, std::ostream& os)
    {
        os << p.layer.size() << std::endl;

        for (auto& ell: p.layer) {
            save_blstm_feat_param(ell, os);
        }
    }

    void save_dblstm_feat_param(dblstm_feat_param_t const& p, std::string filename)
    {
        std::ofstream ofs { filename };
        save_dblstm_feat_param(p, ofs);
    }

    void const_step_update_momentum(dblstm_feat_param_t& p, dblstm_feat_param_t const& grad,
        dblstm_feat_param_t& opt_data, double momentum, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            const_step_update_momentum(p.layer[i], grad.layer[i],
                opt_data.layer[i], momentum, step_size);
        }
    }

    void adagrad_update(dblstm_feat_param_t& p, dblstm_feat_param_t const& grad,
        dblstm_feat_param_t& opt_data, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            adagrad_update(p.layer[i], grad.layer[i],
                opt_data.layer[i], step_size);
        }
    }

    void rmsprop_update(dblstm_feat_param_t& p, dblstm_feat_param_t const& grad,
        dblstm_feat_param_t& opt_data, double decay, double step_size)
    {
        for (int i = 0; i < p.layer.size(); ++i) {
            rmsprop_update(p.layer[i], grad.layer[i],
                opt_data.layer[i], decay, step_size);
        }
    }

    dblstm_feat_nn_t make_dblstm_feat_nn(autodiff::computation_graph& graph,
        dblstm_feat_param_t const& p,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        dblstm_feat_nn_t result;

        for (int i = 0; i < p.layer.size(); ++i) {
            if (i == 0) {
                result.layer.push_back(make_blstm_feat_nn(graph, p.layer[0], inputs));
            } else {
                result.layer.push_back(make_blstm_feat_nn(graph, p.layer[i],
                    result.layer[i-1].output));
            }
        }

        return result;
    }

    dblstm_feat_param_t copy_dblstm_feat_grad(dblstm_feat_nn_t const& nn)
    {
        dblstm_feat_param_t result;

        for (int i = 0; i < nn.layer.size(); ++i) {
            result.layer.push_back(copy_blstm_feat_grad(nn.layer[i]));
        }

        return result;
    }

    void apply_random_mask(dblstm_feat_nn_t& nn, dblstm_feat_param_t const& param,
        std::default_random_engine& gen, double prob)
    {
        std::bernoulli_distribution bernoulli { prob };

        for (int ell = 1; ell < nn.layer.size(); ++ell) {
            la::vector<double> mask_vec;
            mask_vec.resize(param.layer[ell].forward_param.hidden_input.cols());

            for (int i = 0; i < mask_vec.size(); ++i) {
                mask_vec(i) = bernoulli(gen);
            }

            auto& f_mask = nn.layer[ell].forward_feat_nn.input_mask;
            f_mask->output = std::make_shared<la::vector<double>>(mask_vec);

            for (int i = 0; i < mask_vec.size(); ++i) {
                mask_vec(i) = bernoulli(gen);
            }

            auto& b_mask = nn.layer[ell].backward_feat_nn.input_mask;
            b_mask->output = std::make_shared<la::vector<double>>(mask_vec);
        }
    }

    void apply_mask(dblstm_feat_nn_t& nn, dblstm_feat_param_t const& param, double prob)
    {
        for (int ell = 1; ell < nn.layer.size(); ++ell) {
            la::vector<double> mask_vec;
            mask_vec.resize(param.layer[ell].forward_param.hidden_input.cols(), prob);

            auto& f_mask = nn.layer[ell].forward_feat_nn.input_mask;
            f_mask->output = std::make_shared<la::vector<double>>(mask_vec);

            auto& b_mask = nn.layer[ell].backward_feat_nn.input_mask;
            b_mask->output = std::make_shared<la::vector<double>>(mask_vec);
        }
    }
}
