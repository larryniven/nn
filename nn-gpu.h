#ifndef NN_GPU_H
#define NN_GPU_H

#include "nn/nn.h"
#include "la/la-gpu.h"

namespace nn {

    namespace gpu {

        struct param_t {
            param_t();
            param_t(nn::param_t p);

            std::vector<la::gpu::matrix<double>> weight;
            std::vector<la::gpu::vector<double>> bias;
            la::gpu::matrix<double> label_weight;
            la::gpu::vector<double> label_bias;
        };

        nn::param_t to_host(param_t const& p);

        void iadd(param_t& p, param_t const& q);
        void resize_as(param_t& p, param_t const& q);

        void zero_param(param_t& p);

        struct opt_t {
            opt_t();
            opt_t(nn::opt_t o);

            int time;
            param_t first_moment;
            param_t second_moment;
        };

        nn::opt_t to_host(opt_t const& o);

        nn::nn_t make_nn(param_t const& p);
        nn::nn_t make_nn2(param_t const& p);

        void adagrad_update(param_t& p, param_t const& grad,
            opt_t& opt_data, double step_size);

        void adam_update(param_t& p, param_t const& grad,
            opt_t& opt_data, double step_size);

        void move_param(param_t& p, nn_t& nn);
        void move_param(nn_t& nn, param_t& p);
        param_t copy_grad(nn_t const& nn);
        void move_grad(param_t& p, nn_t const& nn);
        void move_grad(nn_t& nn, param_t& p);
        void zero_grad(nn_t& nn);

        struct log_loss {
        
            la::gpu::vector<double> pred;
            la::gpu::vector<double> gold;
        
            log_loss(la::gpu::vector<double> const& pred,
                la::gpu::vector<double> const& gold);
        
            double loss();
        
            la::gpu::vector<double> grad();
        
        };
    }
}

#endif
