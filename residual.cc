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
        std::shared_ptr<autodiff::op_t> cell,
        unit_param_t& param)
    {
        nn_unit_t result;

        result.weight1 = graph.var(la::weak_matrix<double>(param.weight1));
        result.bias1 = graph.var(la::weak_vector<double>(param.bias1));
        result.weight2 = graph.var(la::weak_matrix<double>(param.weight2));
        result.bias2 = graph.var(la::weak_vector<double>(param.bias2));
        // result.input_weight = graph.var(param.input_weight);
        // result.input_bias = graph.var(param.input_bias);

        result.cell = cell;

        std::shared_ptr<autodiff::op_t> h = autodiff::add(
            autodiff::mul(result.weight1, autodiff::relu(cell)), result.bias1);
        result.output = autodiff::add(cell,
            autodiff::add(autodiff::mul(result.weight2, autodiff::relu(h)), result.bias2));

        return result;
    }

    void unit_nn_tie_grad(nn_unit_t& nn, unit_param_t& grad)
    {
        nn.weight1->grad = std::make_shared<la::weak_matrix<double>>(grad.weight1);
        nn.bias1->grad = std::make_shared<la::weak_vector<double>>(grad.bias1);
        nn.weight2->grad = std::make_shared<la::weak_matrix<double>>(grad.weight2);
        nn.bias2->grad = std::make_shared<la::weak_vector<double>>(grad.bias2);
    }

    void resize_as(unit_param_t& p1, unit_param_t const& p2)
    {
        p1.weight1.resize(p2.weight1.rows(), p2.weight1.cols());
        p1.bias1.resize(p2.bias1.size());
        p1.weight2.resize(p2.weight2.rows(), p2.weight2.cols());
        p1.bias2.resize(p2.bias2.size());
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

        ebt::json::json_parser<la::matrix<double>> mat_parser;
        ebt::json::json_parser<la::vector<double>> vec_parser;

        result.input_weight = mat_parser.parse(is);
        std::getline(is, line);
        result.input_bias = vec_parser.parse(is);
        std::getline(is, line);

        std::getline(is, line);
        int layers = std::stoi(line);

        for (int i = 0; i < layers; ++i) {
            result.layer.push_back(load_unit_param(is));
        }

        return result;
    }

    void save_nn_param(nn_param_t const& p, std::ostream& os)
    {
        ebt::json::dump(p.input_weight, os);
        os << std::endl;
        ebt::json::dump(p.input_bias, os);
        os << std::endl;

        os << p.layer.size() << std::endl;

        for (int i = 0; i < p.layer.size(); ++i) {
            save_unit_param(p.layer[i], os);
        }
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

    nn_t make_nn(autodiff::computation_graph& graph,
        nn_param_t& param)
    {
        nn_t result;

        result.input = graph.var();
        result.input_weight = graph.var(param.input_weight);
        result.input_bias = graph.var(param.input_bias);

        std::shared_ptr<autodiff::op_t> cell = autodiff::add(
            autodiff::mul(result.input_weight, result.input), result.input_bias);

        for (int i = 0; i < param.layer.size(); ++i) {
            result.layer.push_back(make_unit_nn(graph, cell, param.layer[i]));
            cell = result.layer.back().output;
        }

        return result;
    }

    void nn_tie_grad(nn_t& nn, nn_param_t& grad)
    {
        nn.input_weight->grad = std::make_shared<la::weak_matrix<double>>(grad.input_weight);
        nn.input_bias->grad = std::make_shared<la::weak_vector<double>>(grad.input_bias);

        for (int i = 0; i < grad.layer.size(); ++i) {
            unit_nn_tie_grad(nn.layer[i], grad.layer[i]);
        }
    }

    void resize_as(nn_param_t& p1, nn_param_t const& p2)
    {
        p1.input_weight.resize(p2.input_weight.rows(), p2.input_weight.cols());
        p1.input_bias.resize(p2.input_bias.size());

        p1.layer.resize(p2.layer.size());
        for (int i = 0; i < p2.layer.size(); ++i) {
            resize_as(p1.layer[i], p2.layer[i]);
        }
    }

    nn_param_t copy_nn_grad(nn_t const& nn)
    {
        nn_param_t result;

        result.input_weight = autodiff::get_grad<la::matrix<double>>(nn.input_weight);
        result.input_bias = autodiff::get_grad<la::vector<double>>(nn.input_bias);

        for (int i = 0; i < nn.layer.size(); ++i) {
            result.layer.push_back(copy_unit_grad(nn.layer[i]));
        }

        return result;
    }

}
