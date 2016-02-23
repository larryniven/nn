#ifndef RNN_GPU_H
#define RNN_GPU_H

#include "la/la-gpu.h"
#include "nn/rnn.h"
#include "autodiff/autodiff-gpu.h"

namespace lstm {

    namespace gpu {

        void bound(la::gpu::vector_like<double>& p, double min, double max);
        void bound(la::gpu::matrix_like<double>& p, double min, double max);

        struct lstm_feat_param_t {
            la::gpu::matrix<double> hidden_input;
            la::gpu::matrix<double> hidden_output;
            la::gpu::vector<double> hidden_bias;

            la::gpu::matrix<double> input_input;
            la::gpu::matrix<double> input_output;
            la::gpu::vector<double> input_peep;
            la::gpu::vector<double> input_bias;

            la::gpu::matrix<double> output_input;
            la::gpu::matrix<double> output_output;
            la::gpu::vector<double> output_peep;
            la::gpu::vector<double> output_bias;

            la::gpu::matrix<double> forget_input;
            la::gpu::matrix<double> forget_output;
            la::gpu::vector<double> forget_peep;
            la::gpu::vector<double> forget_bias;
        };

        lstm::lstm_feat_param_t to_host(lstm_feat_param_t const& param);
        lstm_feat_param_t to_device(lstm::lstm_feat_param_t const& param);
        void resize_as(lstm_feat_param_t& a, lstm_feat_param_t const& b);
        void zero(lstm_feat_param_t& p);

        void bound(lstm_feat_param_t& p, double min, double max);

        void adagrad_update(lstm_feat_param_t& p, lstm_feat_param_t const& grad,
            lstm_feat_param_t& opt_data, double step_size);

        void const_step_update_momentum(lstm_feat_param_t& p, lstm_feat_param_t const& grad,
            lstm_feat_param_t& opt_data, double momentum, double step_size);

        lstm::lstm_feat_nn_t make_forward_lstm_feat_nn(autodiff::computation_graph& g,
            autodiff::gpu::memory_pool<double>& mem,
            lstm_feat_param_t& p,
            std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

        lstm::lstm_feat_nn_t make_backward_lstm_feat_nn(autodiff::computation_graph& g,
            autodiff::gpu::memory_pool<double>& mem,
            lstm_feat_param_t& p,
            std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

        void attach_grad(lstm_feat_param_t& grad, lstm::lstm_feat_nn_t const& nn);

        struct blstm_feat_param_t {
            lstm_feat_param_t forward_param;
            lstm_feat_param_t backward_param;

            la::gpu::matrix<double> forward_output_weight;
            la::gpu::matrix<double> backward_output_weight;
            la::gpu::vector<double> output_bias;
        };

        lstm::blstm_feat_param_t to_host(blstm_feat_param_t const& param);
        blstm_feat_param_t to_host(lstm::blstm_feat_param_t const& param);
        void resize_as(blstm_feat_param_t& a, blstm_feat_param_t const& b);
        void zero(blstm_feat_param_t& p);

        void bound(blstm_feat_param_t& p, double min, double max);

        void adagrad_update(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
            blstm_feat_param_t& opt_data, double step_size);

        void const_step_update_momentum(blstm_feat_param_t& p, blstm_feat_param_t const& grad,
            blstm_feat_param_t& opt_data, double momentum, double step_size);

        lstm::blstm_feat_nn_t make_blstm_feat_nn(autodiff::computation_graph& g,
            autodiff::gpu::memory_pool<double>& mem,
            blstm_feat_param_t& p,
            std::vector<std::shared_ptr<autodiff::op_t>> const& inputs);

        void attach_grad(dblstm_param_t& grad, lstm::blstm_feat_nn_t const& nn);

        struct dblstm_param_t {
            std::vector<blstm_feat_param_t> layer;

            la::gpu::matrix<double> softmax_weight;
            la::gpu::vector<double> softmax_bias;
        };

        lstm::dblstm_param_t to_host(dblstm_param_t const& param);
        dblstm_param_t to_device(lstm::dblstm_param_t const& param);
        void resize_as(dblstm_param_t& a, dblstm_param_t const& b);
        void zero(dblstm_param_t& p);

        void bound(dblstm_param_t& p, double min, double max);

        void adagrad_update(dblstm_param_t& p, dblstm_param_t const& grad,
            dblstm_param_t& opt_data, double step_size);

        void const_step_update_momentum(dblstm_param_t& p, dblstm_param_t const& grad,
            dblstm_param_t& opt_data, double momentum, double step_size);

        struct dblstm_nn_t {
            autodiff::computation_graph graph;

            autodiff::gpu::memory_pool<double> *mem;

            std::vector<blstm_feat_nn_t> layer;

            std::shared_ptr<autodiff::op_t> softmax_weight;
            std::shared_ptr<autodiff::op_t> softmax_bias;

            std::vector<std::shared_ptr<autodiff::op_t>> logprob;
        };

        dblstm_nn_t make_dblstm_nn(dblstm_param_t& p,
            autodiff::gpu::memory_pool<double>& mem,
            std::vector<std::vector<double>> const& frames);

        void attach_grad(dblstm_param_t& grad, dblstm_nn_t const& nn);

        void eval(dblstm_nn_t const& nn);
        void grad(dblstm_nn_t const& nn);

        struct log_loss {

            la::gpu::weak_vector<double> gold;
            la::gpu::weak_vector<double> pred;

            double loss();
        };

    }

}

#endif
