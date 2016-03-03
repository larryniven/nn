#include "nn/lstm-gpu.h"
#include "opt/opt-gpu.h"
#include "autodiff/autodiff-gpu.h"
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>

namespace lstm {

    namespace gpu {

        lstm::lstm_feat_param_t to_host(lstm_feat_param_t const& param)
        {
            lstm::lstm_feat_param_t result;

            result.hidden_input = la::gpu::to_host(param.hidden_input);
            result.hidden_output = la::gpu::to_host(param.hidden_output);
            result.hidden_bias = la::gpu::to_host(param.hidden_bias);

            result.input_input = la::gpu::to_host(param.input_input);
            result.input_output = la::gpu::to_host(param.input_output);
            result.input_peep = la::gpu::to_host(param.input_peep);
            result.input_bias = la::gpu::to_host(param.input_bias);

            result.output_input = la::gpu::to_host(param.output_input);
            result.output_output = la::gpu::to_host(param.output_output);
            result.output_peep = la::gpu::to_host(param.output_peep);
            result.output_bias = la::gpu::to_host(param.output_bias);

            result.forget_input = la::gpu::to_host(param.forget_input);
            result.forget_output = la::gpu::to_host(param.forget_output);
            result.forget_peep = la::gpu::to_host(param.forget_peep);
            result.forget_bias = la::gpu::to_host(param.forget_bias);

            return result;
        }

        lstm_feat_param_t to_device(lstm::lstm_feat_param_t const& param)
        {
            lstm_feat_param_t result;

            result.hidden_input = la::gpu::matrix<double>(param.hidden_input);
            result.hidden_output = la::gpu::matrix<double>(param.hidden_output);
            result.hidden_bias = la::gpu::vector<double>(param.hidden_bias);

            result.input_input = la::gpu::matrix<double>(param.input_input);
            result.input_output = la::gpu::matrix<double>(param.input_output);
            result.input_peep = la::gpu::vector<double>(param.input_peep);
            result.input_bias = la::gpu::vector<double>(param.input_bias);

            result.output_input = la::gpu::matrix<double>(param.output_input);
            result.output_output = la::gpu::matrix<double>(param.output_output);
            result.output_peep = la::gpu::vector<double>(param.output_peep);
            result.output_bias = la::gpu::vector<double>(param.output_bias);

            result.forget_input = la::gpu::matrix<double>(param.forget_input);
            result.forget_output = la::gpu::matrix<double>(param.forget_output);
            result.forget_peep = la::gpu::vector<double>(param.forget_peep);
            result.forget_bias = la::gpu::vector<double>(param.forget_bias);

            return result;
        }

        void resize_as(lstm_feat_param_t& a, lstm_feat_param_t const& b)
        {
            a.hidden_input.resize(b.hidden_input.rows(), b.hidden_input.cols());
            a.hidden_output.resize(b.hidden_output.rows(), b.hidden_output.cols());
            a.hidden_bias.resize(b.hidden_bias.size());

            a.input_input.resize(b.input_input.rows(), b.input_input.cols());
            a.input_output.resize(b.input_output.rows(), b.input_output.cols());
            a.input_peep.resize(b.input_peep.size());
            a.input_bias.resize(b.input_bias.size());

            a.output_input.resize(b.output_input.rows(), b.output_input.cols());
            a.output_output.resize(b.output_output.rows(), b.output_output.cols());
            a.output_peep.resize(b.output_peep.size());
            a.output_bias.resize(b.output_bias.size());

            a.forget_input.resize(b.forget_input.rows(), b.forget_input.cols());
            a.forget_output.resize(b.forget_output.rows(), b.forget_output.cols());
            a.forget_peep.resize(b.forget_peep.size());
            a.forget_bias.resize(b.forget_bias.size());
        }

        void zero(lstm_feat_param_t& p)
        {
            la::gpu::zero(p.hidden_input);
            la::gpu::zero(p.hidden_output);
            la::gpu::zero(p.hidden_bias);

            la::gpu::zero(p.input_input);
            la::gpu::zero(p.input_output);
            la::gpu::zero(p.input_peep);
            la::gpu::zero(p.input_bias);

            la::gpu::zero(p.output_input);
            la::gpu::zero(p.output_output);
            la::gpu::zero(p.output_peep);
            la::gpu::zero(p.output_bias);

            la::gpu::zero(p.forget_input);
            la::gpu::zero(p.forget_output);
            la::gpu::zero(p.forget_peep);
            la::gpu::zero(p.forget_bias);
        }

        void adagrad_update(lstm_feat_param_t& p, lstm_feat_param_t const& grad,
            lstm_feat_param_t& opt_data, double step_size)
        {
            opt::gpu::adagrad_update(p.hidden_input, grad.hidden_input,
                opt_data.hidden_input, step_size);
            opt::gpu::adagrad_update(p.hidden_output, grad.hidden_output,
                opt_data.hidden_output, step_size);
            opt::gpu::adagrad_update(p.hidden_bias, grad.hidden_bias,
                opt_data.hidden_bias, step_size);

            opt::gpu::adagrad_update(p.input_input, grad.input_input,
                opt_data.input_input, step_size);
            opt::gpu::adagrad_update(p.input_output, grad.input_output,
                opt_data.input_output, step_size);
            opt::gpu::adagrad_update(p.input_peep, grad.input_peep,
                opt_data.input_peep, step_size);
            opt::gpu::adagrad_update(p.input_bias, grad.input_bias,
                opt_data.input_bias, step_size);

            opt::gpu::adagrad_update(p.output_input, grad.output_input,
                opt_data.output_input, step_size);
            opt::gpu::adagrad_update(p.output_output, grad.output_output,
                opt_data.output_output, step_size);
            opt::gpu::adagrad_update(p.output_peep, grad.output_peep,
                opt_data.output_peep, step_size);
            opt::gpu::adagrad_update(p.output_bias, grad.output_bias,
                opt_data.output_bias, step_size);

            opt::gpu::adagrad_update(p.forget_input, grad.forget_input,
                opt_data.forget_input, step_size);
            opt::gpu::adagrad_update(p.forget_output, grad.forget_output,
                opt_data.forget_output, step_size);
            opt::gpu::adagrad_update(p.forget_peep, grad.forget_peep,
                opt_data.forget_peep, step_size);
            opt::gpu::adagrad_update(p.forget_bias, grad.forget_bias,
                opt_data.forget_bias, step_size);
        }

        void const_step_update_momentum(lstm_feat_param_t& p, lstm_feat_param_t const& grad,
            lstm_feat_param_t& opt_data, double momentum, double step_size)
        {
            opt::gpu::const_step_update_momentum(p.hidden_input, grad.hidden_input,
                opt_data.hidden_input, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.hidden_output, grad.hidden_output,
                opt_data.hidden_output, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.hidden_bias, grad.hidden_bias,
                opt_data.hidden_bias, momentum, step_size);

            opt::gpu::const_step_update_momentum(p.input_input, grad.input_input,
                opt_data.input_input, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.input_output, grad.input_output,
                opt_data.input_output, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.input_peep, grad.input_peep,
                opt_data.input_peep, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.input_bias, grad.input_bias,
                opt_data.input_bias, momentum, step_size);

            opt::gpu::const_step_update_momentum(p.output_input, grad.output_input,
                opt_data.output_input, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.output_output, grad.output_output,
                opt_data.output_output, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.output_peep, grad.output_peep,
                opt_data.output_peep, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.output_bias, grad.output_bias,
                opt_data.output_bias, momentum, step_size);

            opt::gpu::const_step_update_momentum(p.forget_input, grad.forget_input,
                opt_data.forget_input, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.forget_output, grad.forget_output,
                opt_data.forget_output, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.forget_peep, grad.forget_peep,
                opt_data.forget_peep, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.forget_bias, grad.forget_bias,
                opt_data.forget_bias, momentum, step_size);
        }

        lstm::lstm_feat_nn_t make_forward_lstm_feat_nn(autodiff::computation_graph& g,
            autodiff::gpu::memory_pool<double>& mem,
            lstm_feat_param_t& p,
            std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
        {
            lstm_feat_nn_t result;

            result.hidden_input = g.var(la::gpu::weak_matrix<double>(p.hidden_input));
            result.hidden_output = g.var(la::gpu::weak_matrix<double>(p.hidden_output));
            result.hidden_bias = g.var(la::gpu::weak_vector<double>(p.hidden_bias));

            result.input_input = g.var(la::gpu::weak_matrix<double>(p.input_input));
            result.input_output = g.var(la::gpu::weak_matrix<double>(p.input_output));
            result.input_peep = g.var(la::gpu::weak_vector<double>(p.input_peep));
            result.input_bias = g.var(la::gpu::weak_vector<double>(p.input_bias));

            result.output_input = g.var(la::gpu::weak_matrix<double>(p.output_input));
            result.output_output = g.var(la::gpu::weak_matrix<double>(p.output_output));
            result.output_peep = g.var(la::gpu::weak_vector<double>(p.output_peep));
            result.output_bias = g.var(la::gpu::weak_vector<double>(p.output_bias));

            result.forget_input = g.var(la::gpu::weak_matrix<double>(p.forget_input));
            result.forget_output = g.var(la::gpu::weak_matrix<double>(p.forget_output));
            result.forget_peep = g.var(la::gpu::weak_vector<double>(p.forget_peep));
            result.forget_bias = g.var(la::gpu::weak_vector<double>(p.forget_bias));

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

        lstm::lstm_feat_nn_t make_backward_lstm_feat_nn(autodiff::computation_graph& g,
            autodiff::gpu::memory_pool<double>& mem,
            lstm_feat_param_t& p,
            std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> rev_inputs = inputs;
            std::reverse(rev_inputs.begin(), rev_inputs.end());

            lstm_feat_nn_t result = make_forward_lstm_feat_nn(g, mem, p, rev_inputs);

            std::reverse(result.cell.begin(), result.cell.end());
            std::reverse(result.hidden.begin(), result.hidden.end());
            std::reverse(result.input_gate.begin(), result.input_gate.end());
            std::reverse(result.output_gate.begin(), result.output_gate.end());
            std::reverse(result.forget_gate.begin(), result.forget_gate.end());
            std::reverse(result.output.begin(), result.output.end());

            return result;
        }

        void attach_grad(lstm_feat_param_t& grad, lstm::lstm_feat_nn_t const& nn)
        {
            nn.hidden_input->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.hidden_input));
            nn.hidden_output->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.hidden_output));
            nn.hidden_bias->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.hidden_bias));

            nn.input_input->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.input_input));
            nn.input_output->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.input_output));
            nn.input_peep->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.input_peep));
            nn.input_bias->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.input_bias));

            nn.output_input->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.output_input));
            nn.output_output->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.output_output));
            nn.output_peep->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.output_peep));
            nn.output_bias->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.output_bias));

            nn.forget_input->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.forget_input));
            nn.forget_output->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.forget_output));
            nn.forget_peep->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.forget_peep));
            nn.forget_bias->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.forget_bias));
        }

        lstm::blstm_feat_param_t to_host(blstm_feat_param_t const& param)
        {
            lstm::blstm_feat_param_t result;

            result.forward_param = to_host(param.forward_param);
            result.backward_param = to_host(param.backward_param);

            result.forward_output_weight = la::gpu::to_host(param.forward_output_weight);
            result.backward_output_weight = la::gpu::to_host(param.backward_output_weight);
            result.output_bias = la::gpu::to_host(param.output_bias);

            return result;
        }

        blstm_feat_param_t to_device(lstm::blstm_feat_param_t const& param)
        {
            blstm_feat_param_t result;

            result.forward_param = to_device(param.forward_param);
            result.backward_param = to_device(param.backward_param);

            result.forward_output_weight = la::gpu::matrix<double>(
                param.forward_output_weight);
            result.backward_output_weight = la::gpu::matrix<double>(
                param.backward_output_weight);
            result.output_bias = la::gpu::vector<double>(param.output_bias);

            return result;
        }

        void resize_as(blstm_feat_param_t& a, blstm_feat_param_t const& b)
        {
            resize_as(a.forward_param, b.forward_param);
            resize_as(a.backward_param, b.backward_param);

            a.forward_output_weight.resize(
                b.forward_output_weight.rows(), b.forward_output_weight.cols());
            a.backward_output_weight.resize(
                b.backward_output_weight.rows(), b.backward_output_weight.cols());
            a.output_bias.resize(b.output_bias.size());
        }

        void zero(blstm_feat_param_t& p)
        {
            zero(p.forward_param);
            zero(p.backward_param);

            la::gpu::zero(p.forward_output_weight);
            la::gpu::zero(p.backward_output_weight);
            la::gpu::zero(p.output_bias);
        }

        void adagrad_update(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
            blstm_feat_param_t& opt_data, double step_size)
        {
            adagrad_update(p.forward_param, grad.forward_param,
                opt_data.forward_param, step_size);
            adagrad_update(p.backward_param, grad.backward_param,
                opt_data.backward_param, step_size);

            opt::gpu::adagrad_update(p.forward_output_weight, grad.forward_output_weight,
                opt_data.forward_output_weight, step_size);
            opt::gpu::adagrad_update(p.backward_output_weight, grad.backward_output_weight,
                opt_data.backward_output_weight, step_size);
            opt::gpu::adagrad_update(p.output_bias, grad.output_bias,
                opt_data.output_bias, step_size);
        }

        void const_step_update_momentum(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
            blstm_feat_param_t& opt_data, double momentum, double step_size)
        {
            const_step_update_momentum(p.forward_param, grad.forward_param,
                opt_data.forward_param, momentum, step_size);
            const_step_update_momentum(p.backward_param, grad.backward_param,
                opt_data.backward_param, momentum, step_size);

            opt::gpu::const_step_update_momentum(p.forward_output_weight, grad.forward_output_weight,
                opt_data.forward_output_weight, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.backward_output_weight, grad.backward_output_weight,
                opt_data.backward_output_weight, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.output_bias, grad.output_bias,
                opt_data.output_bias, momentum, step_size);
        }

        lstm::blstm_feat_nn_t make_blstm_feat_nn(autodiff::computation_graph& g,
            autodiff::gpu::memory_pool<double>& mem,
            blstm_feat_param_t& p,
            std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
        {
            blstm_feat_nn_t result;

            result.forward_feat_nn = make_forward_lstm_feat_nn(g, mem, p.forward_param, inputs);
            result.backward_feat_nn = make_backward_lstm_feat_nn(g, mem, p.backward_param, inputs);

            result.forward_output_weight = g.var(
                la::gpu::weak_matrix<double>(p.forward_output_weight));
            result.backward_output_weight = g.var(
                la::gpu::weak_matrix<double>(p.backward_output_weight));
            result.output_bias = g.var(
                la::gpu::weak_vector<double>(p.output_bias));

            for (int i = 0; i < result.forward_feat_nn.output.size(); ++i) {
                result.output.push_back(autodiff::add(
                    std::vector<std::shared_ptr<autodiff::op_t>> {
                        autodiff::mul(result.forward_output_weight,
                            result.forward_feat_nn.output[i]),
                        autodiff::mul(result.backward_output_weight,
                            result.backward_feat_nn.output[i]),
                        result.output_bias
                    }));
            }

            return result;
        }

        void attach_grad(blstm_feat_param_t& grad, lstm::blstm_feat_nn_t const& nn)
        {
            attach_grad(grad.forward_param, nn.forward_feat_nn);
            attach_grad(grad.backward_param, nn.backward_feat_nn);

            nn.forward_output_weight->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.forward_output_weight));
            nn.backward_output_weight->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.backward_output_weight));
            nn.output_bias->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.output_bias));
        }

        lstm::dblstm_param_t to_host(dblstm_param_t const& param)
        {
            lstm::dblstm_param_t result;

            for (int i = 0; i < param.layer.size(); ++i) {
                result.layer.push_back(to_host(param.layer[i]));
            }

            result.softmax_weight = la::gpu::to_host(param.softmax_weight);
            result.softmax_bias = la::gpu::to_host(param.softmax_bias);

            return result;
        }

        dblstm_param_t to_device(lstm::dblstm_param_t const& param)
        {
            dblstm_param_t result;

            for (int i = 0; i < param.layer.size(); ++i) {
                result.layer.push_back(to_device(param.layer[i]));
            }

            result.softmax_weight = la::gpu::matrix<double>(param.softmax_weight);
            result.softmax_bias = la::gpu::vector<double>(param.softmax_bias);

            return result;
        }

        void resize_as(dblstm_param_t& a, dblstm_param_t const& b)
        {
            a.layer.resize(b.layer.size());

            for (int i = 0; i < b.layer.size(); ++i) {
                resize_as(a.layer[i], b.layer[i]);
            }

            a.softmax_weight.resize(b.softmax_weight.rows(), b.softmax_weight.cols());
            a.softmax_bias.resize(b.softmax_bias.size());
        }

        void zero(dblstm_param_t& p)
        {
            for (auto& ell: p.layer) {
                zero(ell);
            }

            la::gpu::zero(p.softmax_weight);
            la::gpu::zero(p.softmax_bias);
        }

        void adagrad_update(dblstm_param_t& p, dblstm_param_t const& grad,
            dblstm_param_t& opt_data, double step_size)
        {
            for (int i = 0; i < p.layer.size(); ++i) {
                adagrad_update(p.layer[i], grad.layer[i],
                    opt_data.layer[i], step_size);
            }

            opt::gpu::adagrad_update(p.softmax_weight, grad.softmax_weight,
                opt_data.softmax_weight, step_size);
            opt::gpu::adagrad_update(p.softmax_bias, grad.softmax_bias,
                opt_data.softmax_bias, step_size);
        }

        void const_step_update_momentum(dblstm_param_t& p, dblstm_param_t const& grad,
            dblstm_param_t& opt_data, double momentum, double step_size)
        {
            for (int i = 0; i < p.layer.size(); ++i) {
                const_step_update_momentum(p.layer[i], grad.layer[i],
                    opt_data.layer[i], momentum, step_size);
            }

            opt::gpu::const_step_update_momentum(p.softmax_weight, grad.softmax_weight,
                opt_data.softmax_weight, momentum, step_size);
            opt::gpu::const_step_update_momentum(p.softmax_bias, grad.softmax_bias,
                opt_data.softmax_bias, momentum, step_size);
        }

        dblstm_nn_t make_dblstm_nn(dblstm_param_t& p,
            autodiff::gpu::memory_pool<double>& mem,
            std::vector<std::vector<double>> const& frames)
        {
            dblstm_nn_t result;
            result.mem = &mem;

            unsigned int dim = frames.front().size();

            std::vector<double> frame_block;
            for (auto& f: frames) {
                frame_block.insert(frame_block.end(), f.begin(), f.end());
            }

            double *d = result.mem->alloc(frame_block.size());

            la::gpu::weak_vector<double> frame_device_block(d, frame_block.size());
            la::gpu::to_device(frame_device_block, la::vector<double>(frame_block));

            std::vector<std::shared_ptr<autodiff::op_t>> inputs;

            for (int i = 0; i < frames.size(); ++i) {
                auto v = result.graph.var(la::gpu::weak_vector<double>(
                    frame_device_block.data() + i * dim, dim));
                double *d = result.mem->alloc(dim);
                la::gpu::weak_vector<double> g(d, dim);
                la::gpu::zero(g);
                v->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
                
                inputs.push_back(v);
            }

            for (int i = 0; i < p.layer.size(); ++i) {
                if (i == 0) {
                    result.layer.push_back(make_blstm_feat_nn(
                        result.graph, mem, p.layer[0], inputs));
                } else {
                    result.layer.push_back(make_blstm_feat_nn(
                        result.graph, mem, p.layer[i], result.layer[i-1].output));
                }
            }

            result.softmax_weight = result.graph.var(la::gpu::weak_matrix<double>(p.softmax_weight));
            result.softmax_bias = result.graph.var(la::gpu::weak_vector<double>(p.softmax_bias));

            for (int i = 0; i < result.layer.back().output.size(); ++i) {
                result.logprob.push_back(autodiff::logsoftmax(autodiff::add(
                    autodiff::mul(result.softmax_weight, result.layer.back().output[i]),
                    result.softmax_bias)));
            }

            std::vector<std::shared_ptr<autodiff::op_t>> order = autodiff::topo_order(result.logprob);
            autodiff::gpu::alloc(order, mem, autodiff::gpu::alloc_funcs);

            return result;
        }

        void attach_grad(dblstm_param_t& grad, dblstm_nn_t const& nn)
        {
            for (int i = 0; i < nn.layer.size(); ++i) {
                attach_grad(grad.layer[i], nn.layer[i]);
            }

            nn.softmax_weight->grad = std::make_shared<la::gpu::weak_matrix<double>>(
                la::gpu::weak_matrix<double>(grad.softmax_weight));
            nn.softmax_bias->grad = std::make_shared<la::gpu::weak_vector<double>>(
                la::gpu::weak_vector<double>(grad.softmax_bias));
        }

        void eval(dblstm_nn_t const& nn)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> order
                = autodiff::topo_order(nn.logprob);

            autodiff::eval(order, autodiff::gpu::eval_funcs);
        }

        void grad(dblstm_nn_t const& nn)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> order
                = autodiff::topo_order(nn.logprob);

            autodiff::grad(order, autodiff::gpu::grad_funcs);
        }

        double log_loss::loss()
        {
            return -la::gpu::dot(gold, pred);
        }

    }

}
