#ifndef TENSOR_TREE_GPU_H
#define TENSOR_TREE_GPU_H

#include "nn/tensor-tree.h"
#include "la/la-gpu.h"

namespace tensor_tree {

    namespace gpu {

        void to_device(std::shared_ptr<vertex> t);
        void to_host(std::shared_ptr<vertex> t);

        la::gpu::tensor<double>& get_gpu_tensor(std::shared_ptr<vertex> t);

        void load_tensor(std::shared_ptr<vertex> root, std::istream& is);
        void load_tensor(std::shared_ptr<vertex> root, std::string filename);

        void save_tensor(std::shared_ptr<vertex> root, std::ostream& os);
        void save_tensor(std::shared_ptr<vertex> root, std::string filename);

        void resize_as(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2);

        void axpy(std::shared_ptr<vertex> p1, double a, std::shared_ptr<vertex> p2);

        void zero(std::shared_ptr<vertex> p);

        double norm(std::shared_ptr<vertex> root);

        bool has_nan(std::shared_ptr<vertex> root);

        std::shared_ptr<vertex> deep_copy(std::shared_ptr<vertex> root);

        void print_tree(std::shared_ptr<vertex> root);

        void const_step_update(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            double step_size);

        void const_step_update_momentum(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> opt_data, double momentum, double step_size);

        void adagrad_update(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> accu_grad_sq, double step_size);

        void rmsprop_update(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> opt_data, double decay, double step_size);

        void adam_update(std::shared_ptr<vertex> param,
            std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> first_moment,
            std::shared_ptr<vertex> second_moment,
            int& time, double alpha, double beta1, double beta2);

        struct const_step_opt
            : public tensor_tree::const_step_opt {

            using tensor_tree::const_step_opt::const_step_opt;

            virtual void update(std::shared_ptr<vertex> grad) override;

        };

        struct const_step_momentum_opt
            : public tensor_tree::const_step_momentum_opt {

            using tensor_tree::const_step_momentum_opt::const_step_momentum_opt;

            virtual void update(std::shared_ptr<vertex> grad) override;

        };

        struct adagrad_opt
            : public tensor_tree::adagrad_opt {

            using tensor_tree::adagrad_opt::adagrad_opt;

            virtual void update(std::shared_ptr<vertex> grad) override;

        };

        struct rmsprop_opt
            : public tensor_tree::rmsprop_opt {

            using tensor_tree::rmsprop_opt::rmsprop_opt;

            virtual void update(std::shared_ptr<vertex> grad) override;

        };

        struct adam_opt
            : public tensor_tree::adam_opt {

            using tensor_tree::adam_opt::adam_opt;

            virtual void update(std::shared_ptr<vertex> grad) override;

        };
    }

}

#endif
