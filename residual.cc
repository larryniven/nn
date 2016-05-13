#include "nn/residual.h"
#include "ebt/json.h"

namespace residual {

    unit_param_t load_unit_param(std::istream& is)
    {
        std::string line;
        unit_param_t result;
        ebt::json::json_parser<la::matrix<double>> mat_parser;
        ebt::json::json_parser<la::vector<double>> vec_parser;

        result.weight1 = mat_parser.parse(is);
        std::getline(is, line);
        result.bias1 = vec_parser.parse(is);
        std::getline(is, line);
        result.weight2 = mat_parser.parse(is);
        std::getline(is, line);
        result.bias2 = vec_parser.parse(is);
        std::getline(is, line);

        return result;
    }

    void save_unit_param(unit_param_t const& p, std::ostream& os)
    {
        ebt::json::dump(p.weight1, os);
        os << std::endl;
        ebt::json::dump(p.bias1, os);
        os << std::endl;
        ebt::json::dump(p.weight2, os);
        os << std::endl;
        ebt::json::dump(p.bias2, os);
        os << std::endl;
    }

    void adagrad_update(unit_param_t& param, unit_param_t const& grad,
        unit_param_t& accu_grad_sq, double step_size)
    {
        opt::adagrad_update(param.weight1, grad.weight1, accu_grad_sq.weight1, step_size);
        opt::adagrad_update(param.bias1, grad.bias1, accu_grad_sq.bias1, step_size);
        opt::adagrad_update(param.weight2, grad.weight2, accu_grad_sq.weight2, step_size);
        opt::adagrad_update(param.bias2, grad.bias2, accu_grad_sq.bias2, step_size);
    }

    nn_unit_t make_unit_nn(autodiff::computation_graph& graph,
        unit_param_t const& param)
    {
        nn_unit_t result;

        result.weight1 = graph.var(param.weight1);
        result.bias1 = graph.var(param.bias1);
        result.weight2 = graph.var(param.weight2);
        result.bias2 = graph.var(param.bias2);
        // result.input_weight = graph.var(param.input_weight);
        // result.input_bias = graph.var(param.input_bias);

        std::shared_ptr<autodiff::op_t> h = autodiff::add(
            autodiff::mul(result.weight1, autodiff::relu(result.cell)), result.bias1);
        result.output = autodiff::add(result.cell,
            autodiff::add(autodiff::mul(result.weight2, autodiff::relu(h)), result.bias2));

        return result;
    }

    unit_param_t copy_unit_grad(nn_unit_t const& unit)
    {
        unit_param_t result;

        result.weight1 = autodiff::get_grad<la::matrix<double>>(unit.weight1);
        result.bias1 = autodiff::get_grad<la::vector<double>>(unit.bias1);
        result.weight2 = autodiff::get_grad<la::matrix<double>>(unit.weight2);
        result.bias2 = autodiff::get_grad<la::vector<double>>(unit.bias2);

        return result;
    }

    nn_param_t load_nn_param(std::istream& is)
    {
        nn_param_t result;
        std::string line;

        std::getline(is, line);
        int layers = std::stoi(line);

        for (int i = 0; i < layers; ++i) {
            result.layer.push_back(load_unit_param(is));
        }

        ebt::json::json_parser<la::matrix<double>> mat_parser;
        ebt::json::json_parser<la::vector<double>> vec_parser;

        mat_parser.parse(is);
        std::getline(is, line);
        vec_parser.parse(is);
        std::getline(is, line);

        return result;
    }

    void save_nn_param(nn_param_t const& p, std::ostream& os)
    {
        os << p.layer.size() << std::endl;

        for (int i = 0; i < p.layer.size(); ++i) {
            save_unit_param(p.layer[i], os);
        }

        ebt::json::dump(p.input_weight, os);
        os << std::endl;
        ebt::json::dump(p.input_bias, os);
        os << std::endl;
    }

    void adagrad_update(nn_param_t& param, nn_param_t const& grad,
        nn_param_t& accu_grad_sq, double step_size)
    {
        for (int i = 0; i < param.layer.size(); ++i) {
            adagrad_update(param.layer[i], grad.layer[i], accu_grad_sq.layer[i], step_size);
        }

        opt::adagrad_update(param.input_weight, grad.input_weight, accu_grad_sq.input_weight, step_size);
        opt::adagrad_update(param.input_bias, grad.input_bias, accu_grad_sq.input_bias, step_size);
    }

    nn_param_t copy_nn_grad(nn_t const& nn)
    {
        nn_param_t result;

        for (int i = 0; i < nn.layer.size(); ++i) {
            result.layer.push_back(copy_unit_grad(nn.layer[i]));
        }

        result.input_weight = autodiff::get_grad<la::matrix<double>>(nn.input_weight);
        result.input_bias = autodiff::get_grad<la::vector<double>>(nn.input_bias);

        return result;
    }

}
