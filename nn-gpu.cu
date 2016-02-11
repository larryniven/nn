#include "nn/nn-gpu.h"
#include "opt/opt-gpu.h"

namespace nn {

    namespace gpu {

        param_t::param_t()
        {}

        param_t::param_t(nn::param_t p)
        {
            for (auto& w: p.weight) {
                weight.push_back(la::gpu::matrix<double>(w));
            }

            for (auto& b: p.bias) {
                bias.push_back(la::gpu::vector<double>(b));
            }

            label_weight = la::gpu::matrix<double>(p.label_weight);
            label_bias = la::gpu::vector<double>(p.label_bias);
        }

        nn::param_t to_host(param_t const& p)
        {
            nn::param_t result;

            for (auto& w: p.weight) {
                result.weight.push_back(to_host(w));
            }

            for (auto& b: p.bias) {
                result.bias.push_back(to_host(b));
            }

            result.label_weight = to_host(p.label_weight);
            result.label_bias = to_host(p.label_bias);

            return result;
        }

        void iadd(param_t& p, param_t const& q)
        {
            for (int i = 0; i < p.weight.size(); ++i) {
                la::gpu::iadd(p.weight[i], q.weight[i]);
                la::gpu::iadd(p.bias[i], q.bias[i]);
            }
            la::gpu::iadd(p.label_weight, q.label_weight);
            la::gpu::iadd(p.label_bias, q.label_bias);
        }

        void resize_as(param_t& p, param_t const& q)
        {
            for (int i = 0; i < q.weight.size(); ++i) {
                la::gpu::matrix<double> m;
                m.resize(q.weight[i].rows(), q.weight[i].cols());
                p.weight.push_back(std::move(m));
            }

            for (int i = 0; i < q.bias.size(); ++i) {
                la::gpu::vector<double> v;
                v.resize(q.bias[i].size());
                p.bias.push_back(std::move(v));
            }

            p.label_weight.resize(q.label_weight.rows(), q.label_weight.cols());

            p.label_bias.resize(q.label_bias.size());
        }

        void zero_param(param_t& p)
        {
            for (int i = 0; i < p.weight.size(); ++i) {
                cudaMemset(p.weight[i].data(), 0,
                    p.weight[i].rows() * p.weight[i].cols() * sizeof(double));
                cudaMemset(p.bias[i].data(), 0, p.bias[i].size() * sizeof(double));
            }
            cudaMemset(p.label_weight.data(), 0,
                p.label_weight.rows() * p.label_weight.cols() * sizeof(double));
            cudaMemset(p.label_bias.data(), 0, p.label_bias.size() * sizeof(double));
        }

        opt_t::opt_t()
        {}

        opt_t::opt_t(nn::opt_t o)
            : time(o.time)
            , first_moment(o.first_moment)
            , second_moment(o.second_moment)
        {}

        nn::opt_t to_host(opt_t const& o)
        {
            nn::opt_t result;

            result.time = o.time;
            result.first_moment = to_host(o.first_moment);
            result.second_moment = to_host(o.second_moment);

            return result;
        }

        nn_t make_nn(param_t const& p)
        {
            nn_t nn;

            nn.hidden.push_back(nn.graph.var());

            for (int i = 0; i < p.weight.size(); ++i) {
                auto w_var = nn.graph.var(la::gpu::matrix<double>(p.weight[i]));
                auto b_var = nn.graph.var(la::gpu::vector<double>(p.bias[i]));
                nn.weight.push_back(w_var);
                nn.bias.push_back(b_var);
                nn.hidden.push_back(autodiff::relu(
                    autodiff::add(autodiff::mul(w_var, nn.hidden.back()), b_var)
                ));
            }

            nn.label_weight = nn.graph.var(la::gpu::matrix<double>(p.label_weight));
            nn.label_bias = nn.graph.var(la::gpu::vector<double>(p.label_bias));

            nn.output = autodiff::logsoftmax(autodiff::add(
                autodiff::mul(nn.label_weight, nn.hidden.back()), nn.label_bias));

            return nn;
        }

        nn_t make_nn2(param_t const& p)
        {
            nn_t nn;

            nn.hidden.push_back(nn.graph.var());

            for (int i = 0; i < p.weight.size(); ++i) {
                auto w_var = nn.graph.var(la::gpu::matrix<double>(p.weight[i]));
                auto b_var = nn.graph.var(la::gpu::vector<double>(p.bias[i]));
                nn.weight.push_back(w_var);
                nn.bias.push_back(b_var);
                nn.hidden.push_back(autodiff::relu(
                    autodiff::add(autodiff::mul(w_var, nn.hidden.back()), b_var)
                ));
            }

            nn.label_weight = nn.graph.var(la::gpu::matrix<double>(p.label_weight));
            nn.label_bias = nn.graph.var(la::gpu::vector<double>(p.label_bias));

            std::vector<std::shared_ptr<autodiff::op_t>> hiddens {nn.hidden.begin() + 1, nn.hidden.end()};

            nn.output = autodiff::logsoftmax(autodiff::add(
                autodiff::mul(nn.label_weight, autodiff::add(hiddens)), nn.label_bias));

            return nn;
        }

        void adagrad_update(param_t& p, param_t const& grad,
            opt_t& opt_data, double step_size)
        {
            for (int i = 0; i < p.weight.size(); ++i) {
                opt::gpu::adagrad_update(p.weight[i], grad.weight[i],
                    opt_data.first_moment.weight[i], step_size);
                opt::gpu::adagrad_update(p.bias[i], grad.bias[i],
                    opt_data.first_moment.bias[i], step_size);
            }

            opt::gpu::adagrad_update(p.label_weight, grad.label_weight,
                opt_data.first_moment.label_weight, step_size);
            opt::gpu::adagrad_update(p.label_bias, grad.label_bias,
                opt_data.first_moment.label_bias, step_size);
        }

        void adam_update(param_t& p, param_t const& grad, opt_t& opt_data, double step_size)
        {
            ++opt_data.time;

            for (int i = 0; i < p.weight.size(); ++i) {
                opt::gpu::adam_update(p.weight[i], grad.weight[i],
                    opt_data.first_moment.weight[i], opt_data.second_moment.weight[i],
                    1 + opt_data.time / 1.0e6, step_size, 0.9, 0.999);
                opt::gpu::adam_update(p.bias[i], grad.bias[i],
                    opt_data.first_moment.bias[i], opt_data.second_moment.bias[i],
                    1 + opt_data.time / 1.0e6, step_size, 0.9, 0.999);
            }

            opt::gpu::adam_update(p.label_weight, grad.label_weight,
                opt_data.first_moment.label_weight, opt_data.second_moment.label_weight,
                1 + opt_data.time / 1.0e6, step_size, 0.9, 0.999);
            opt::gpu::adam_update(p.label_bias, grad.label_bias,
                opt_data.first_moment.label_bias, opt_data.second_moment.label_bias,
                1 + opt_data.time / 1.0e6, step_size, 0.9, 0.999);
        }

        void move_param(param_t& result, nn_t& nn)
        {
            for (int i = 0; i < nn.weight.size(); ++i) {
                result.weight[i] = std::move(
                    autodiff::get_output<la::gpu::matrix<double>>(nn.weight[i]));
                result.bias[i] = std::move(
                    autodiff::get_output<la::gpu::vector<double>>(nn.bias[i]));
            }
    
            result.label_weight = std::move(
                autodiff::get_output<la::gpu::matrix<double>>(nn.label_weight));
            result.label_bias = std::move(
                autodiff::get_output<la::gpu::vector<double>>(nn.label_bias));
        }

        void move_param(nn_t& nn, param_t& p)
        {
            for (int i = 0; i < p.weight.size(); ++i) {
                autodiff::get_output<la::gpu::matrix<double>>(nn.weight[i])
                    = std::move(p.weight[i]);
                autodiff::get_output<la::gpu::vector<double>>(nn.bias[i])
                    = std::move(p.bias[i]);
            }
    
            autodiff::get_output<la::gpu::matrix<double>>(nn.label_weight)
                = std::move(p.label_weight);
            autodiff::get_output<la::gpu::vector<double>>(nn.label_bias)
                = std::move(p.label_bias);
        }

        param_t copy_grad(nn_t const& nn)
        {
            param_t result;
    
            for (int i = 0; i < nn.weight.size(); ++i) {
                result.weight.push_back(autodiff::get_grad<la::gpu::matrix<double>>(nn.weight[i]));
                result.bias.push_back(autodiff::get_grad<la::gpu::vector<double>>(nn.bias[i]));
            }
    
            result.label_weight = autodiff::get_grad<la::gpu::matrix<double>>(nn.label_weight);
            result.label_bias = autodiff::get_grad<la::gpu::vector<double>>(nn.label_bias);
    
            return result;
        }

        void move_grad(param_t& result, nn_t const& nn)
        {
            for (int i = 0; i < nn.weight.size(); ++i) {
                result.weight[i] = std::move(
                    autodiff::get_grad<la::gpu::matrix<double>>(nn.weight[i]));
                result.bias[i] = std::move(
                    autodiff::get_grad<la::gpu::vector<double>>(nn.bias[i]));
            }
    
            result.label_weight = std::move(
                autodiff::get_grad<la::gpu::matrix<double>>(nn.label_weight));
            result.label_bias = std::move(
                autodiff::get_grad<la::gpu::vector<double>>(nn.label_bias));
        }

        void move_grad(nn_t& nn, param_t& p)
        {
            for (int i = 0; i < p.weight.size(); ++i) {
                autodiff::get_grad<la::gpu::matrix<double>>(nn.weight[i])
                    = std::move(p.weight[i]);
                autodiff::get_grad<la::gpu::vector<double>>(nn.bias[i])
                    = std::move(p.bias[i]);
            }
    
            autodiff::get_grad<la::gpu::matrix<double>>(nn.label_weight)
                = std::move(p.label_weight);
            autodiff::get_grad<la::gpu::vector<double>>(nn.label_bias)
                = std::move(p.label_bias);
        }

        void zero_grad(nn_t& nn)
        {
            la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(nn.hidden.front()));

            for (int i = 1; i < nn.hidden.size(); ++i) {
                la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(nn.hidden[i]));

                auto add = autodiff::get_child(nn.hidden[i], 0);
                la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(add));

                auto mul = autodiff::get_child(add, 0);
                la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(mul));

                auto bias = autodiff::get_child(add, 1);
                la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(bias));

                auto weight = autodiff::get_child(mul, 0);
                la::gpu::zero(autodiff::get_grad<la::gpu::matrix<double>>(weight));
            }

            la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(nn.output));

            auto add = autodiff::get_child(nn.output, 0);
            la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(add));

            auto mul = autodiff::get_child(add, 0);
            la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(mul));

            auto label_bias = autodiff::get_child(add, 1);
            la::gpu::zero(autodiff::get_grad<la::gpu::vector<double>>(label_bias));

            auto label_weight = autodiff::get_child(mul, 0);
            la::gpu::zero(autodiff::get_grad<la::gpu::matrix<double>>(label_weight));
        }

        log_loss::log_loss(
            la::gpu::vector<double> const& pred, la::gpu::vector<double> const& gold)
            : pred(pred), gold(gold)
        {}
        
        double log_loss::loss()
        {
            return -la::gpu::dot(pred, gold);
        }
        
        la::gpu::vector<double> log_loss::grad()
        {
            return la::gpu::mul(gold, -1);
        }
    }
}
