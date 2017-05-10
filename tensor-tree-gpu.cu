#include "nn/tensor-tree-gpu.h"
#include "opt/opt.h"
#include "opt/opt-gpu.h"

namespace tensor_tree {

    namespace gpu {

        void to_device(std::shared_ptr<vertex> t)
        {
            auto order = leaves_pre_order(t);

            for (auto& t: order) {
                if (t->data == nullptr) {
                    continue;
                }

                if (t->type == "tensor") {
                    la::gpu::tensor<double> dt { get_tensor(t) };
                    t->type = "gpu-tensor";
                    t->data = std::make_shared<la::gpu::tensor<double>>(std::move(dt));
                }
            }
        }

        void to_host(std::shared_ptr<vertex> t)
        {
            auto order = leaves_pre_order(t);

            for (auto& t: order) {
                if (t->data == nullptr) {
                    continue;
                }

                if (t->type == "gpu-tensor") {
                    t->data = std::make_shared<la::tensor<double>>(la::gpu::to_host(get_gpu_tensor(t)));
                    t->type = "tensor";
                }
            }
        }

        la::gpu::tensor<double>& get_gpu_tensor(std::shared_ptr<vertex> t)
        {
            if (t->type == "gpu-tensor") {
                return get_data<la::gpu::tensor<double>>(t);
            } else {
                throw std::logic_error("expecting gpu-tensor");
            }
        }

        void resize_as(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2)
        {
            auto p1_order = leaves_pre_order(p1);
            auto p2_order = leaves_pre_order(p2);

            for (int i = 0; i < p1_order.size(); ++i) {
                if (p1_order[i]->type == "tensor") {
                    la::tensor<double> m;
                    auto& m2 = get_tensor(p2_order[i]);
                    la::resize_as(m, m2);
                    p1_order[i]->data = std::make_shared<la::tensor<double>>(std::move(m));
                } else if (p1_order[i]->type == "gpu-tensor") {
                    la::gpu::tensor<double> m;
                    auto& m2 = get_gpu_tensor(p2_order[i]);
                    la::gpu::resize_as(m, m2);
                    p1_order[i]->data = std::make_shared<la::gpu::tensor<double>>(std::move(m));
                }
            }
        }

        void imul(std::shared_ptr<vertex> root, double a)
        {
            auto order = leaves_pre_order(root);

            for (auto& t: order) {
                if (t->data == nullptr) {
                    continue;
                }

                if (t->type == "tensor") {
                    la::imul(get_tensor(t), a);
                } else if (t->type == "gpu-tensor") {
                    la::gpu::imul(get_gpu_tensor(t), a);
                }
            }
        }

        void iadd(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2)
        {
            auto p1_order = leaves_pre_order(p1);
            auto p2_order = leaves_pre_order(p2);

            for (int i = 0; i < p1_order.size(); ++i) {
                if (p2_order[i]->data == nullptr) {
                    continue;
                }

                if (p1_order[i]->type == "tensor") {
                    la::iadd(get_tensor(p1_order[1]), get_tensor(p2_order[i]));
                } else if (p1_order[i]->type == "gpu-tensor") {
                    la::gpu::iadd(get_gpu_tensor(p1_order[1]), get_gpu_tensor(p2_order[i]));
                }
            }
        }

        void isub(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2)
        {
            auto p1_order = leaves_pre_order(p1);
            auto p2_order = leaves_pre_order(p2);

            for (int i = 0; i < p1_order.size(); ++i) {
                if (p2_order[i]->data == nullptr) {
                    continue;
                }

                if (p1_order[i]->type == "tensor") {
                    la::isub(get_tensor(p1_order[i]), get_tensor(p2_order[i]));
                } else if (p1_order[i]->type == "gpu-tensor") {
                    la::gpu::isub(get_gpu_tensor(p1_order[i]), get_gpu_tensor(p2_order[i]));
                }
            }
        }

        void zero(std::shared_ptr<vertex> root)
        {
            auto order = leaves_pre_order(root);

            for (auto& t: order) {
                if (t->data == nullptr) {
                    continue;
                }

                if (t->type == "tensor") {
                    la::zero(get_tensor(t));
                } else if (t->type == "gpu-tensor") {
                    la::gpu::zero(get_gpu_tensor(t));
                }
            }
        }

        double norm(std::shared_ptr<vertex> root)
        {
            auto order = leaves_pre_order(root);

            double result = 0;

            for (int i = 0; i < order.size(); ++i) {
                if (order[i]->data == nullptr) {
                    continue;
                }

                if (order[i]->type == "tensor") {
                    result += std::pow(la::norm(get_tensor(order[i])), 2);
                } else if (order[i]->type == "gpu-tensor") {
                    result += std::pow(la::gpu::norm(get_gpu_tensor(order[i])), 2);
                }
            }

            return std::sqrt(result);
        }

        bool has_nan(std::shared_ptr<vertex> root)
        {
            auto order = leaves_pre_order(root);

            for (int i = 0; i < order.size(); ++i) {
                if (order[i]->data == nullptr) {
                    continue;
                }

                if (order[i]->type == "tensor") {
                    if (la::has_nan(get_tensor(order[i]))) {
                        return true;
                    }
                } else if (order[i]->type == "gpu-tensor") {
                    if (la::gpu::has_nan(get_gpu_tensor(order[i]))) {
                        return true;
                    }
                }
            }

            return false;
        }

        void const_step_update(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            double step_size)
        {
            auto param_order = leaves_pre_order(param);
            auto grad_order = leaves_pre_order(grad);

            assert(param_order.size() == grad_order.size());

            for (int i = 0; i < param_order.size(); ++i) {
                if (grad_order[i]->data == nullptr) {
                    continue;
                }

                if (param_order[i]->type == "tensor") {
                    opt::const_step_update(
                        get_tensor(param_order[i]),
                        get_tensor(grad_order[i]),
                        step_size);
                } else if (param_order[i]->type == "gpu-tensor") {
                    opt::gpu::const_step_update(
                        get_gpu_tensor(param_order[i]),
                        get_gpu_tensor(grad_order[i]),
                        step_size);
                }
            }
        }

        void const_step_update_momentum(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> opt_data, double momentum, double step_size)
        {
            auto param_order = leaves_pre_order(param);
            auto grad_order = leaves_pre_order(grad);
            auto opt_data_order = leaves_pre_order(opt_data);

            assert(param_order.size() == grad_order.size()
                && grad_order.size() == opt_data_order.size());

            for (int i = 0; i < param_order.size(); ++i) {
                if (grad_order[i]->data == nullptr) {
                    continue;
                }

                if (param_order[i]->type == "tensor") {
                    opt::const_step_update_momentum(
                        get_tensor(param_order[i]),
                        get_tensor(grad_order[i]),
                        get_tensor(opt_data_order[i]),
                        momentum, step_size);
                } else if (param_order[i]->type == "gpu-tensor") {
                    opt::gpu::const_step_update_momentum(
                        get_gpu_tensor(param_order[i]),
                        get_gpu_tensor(grad_order[i]),
                        get_gpu_tensor(opt_data_order[i]),
                        momentum, step_size);
                }
            }
        }

        void adagrad_update(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> accu_grad_sq, double step_size)
        {
            auto param_order = leaves_pre_order(param);
            auto grad_order = leaves_pre_order(grad);
            auto accu_grad_sq_order = leaves_pre_order(accu_grad_sq);

            assert(param_order.size() == grad_order.size()
                && grad_order.size() == accu_grad_sq_order.size());

            for (int i = 0; i < param_order.size(); ++i) {
                if (grad_order[i]->data == nullptr) {
                    continue;
                }

                if (param_order[i]->type == "tensor") {
                    opt::adagrad_update(
                        get_tensor(param_order[i]),
                        get_tensor(grad_order[i]),
                        get_tensor(accu_grad_sq_order[i]),
                        step_size);
                } else if (param_order[i]->type == "gpu-tensor") {
                    opt::gpu::adagrad_update(
                        get_gpu_tensor(param_order[i]),
                        get_gpu_tensor(grad_order[i]),
                        get_gpu_tensor(accu_grad_sq_order[i]),
                        step_size);
                }
            }
        }

        void rmsprop_update(std::shared_ptr<vertex> param, std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> opt_data, double decay, double step_size)
        {
            auto param_order = leaves_pre_order(param);
            auto grad_order = leaves_pre_order(grad);
            auto opt_data_order = leaves_pre_order(opt_data);

            assert(param_order.size() == grad_order.size()
                && grad_order.size() == opt_data_order.size());

            for (int i = 0; i < param_order.size(); ++i) {
                if (grad_order[i]->data == nullptr) {
                    continue;
                }

                if (param_order[i]->type == "tensor") {
                    opt::rmsprop_update(
                        get_tensor(param_order[i]),
                        get_tensor(grad_order[i]),
                        get_tensor(opt_data_order[i]),
                        decay, step_size);
                } else if (param_order[i]->type == "gpu-tensor") {
                    opt::gpu::rmsprop_update(
                        get_gpu_tensor(param_order[i]),
                        get_gpu_tensor(grad_order[i]),
                        get_gpu_tensor(opt_data_order[i]),
                        decay, step_size);
                }
            }
        }

        void adam_update(std::shared_ptr<vertex> param,
            std::shared_ptr<vertex> grad,
            std::shared_ptr<vertex> first_moment,
            std::shared_ptr<vertex> second_moment,
            int& time, double alpha, double beta1, double beta2)
        {
            auto param_order = leaves_pre_order(param);
            auto grad_order = leaves_pre_order(grad);
            auto first_moment_order = leaves_pre_order(first_moment);
            auto second_moment_order = leaves_pre_order(second_moment);

            assert(param_order.size() == grad_order.size()
                && grad_order.size() == first_moment_order.size()
                && first_moment_order.size() == second_moment_order.size());

            for (int i = 0; i < param_order.size(); ++i) {
                if (grad_order[i]->data == nullptr) {
                    continue;
                }

                if (param_order[i]->type == "tensor") {
                    opt::adam_update(
                        get_tensor(param_order[i]),
                        get_tensor(grad_order[i]),
                        get_tensor(first_moment_order[i]),
                        get_tensor(second_moment_order[i]),
                        time, alpha, beta1, beta2);
                } else if (param_order[i]->type == "gpu-tensor") {
                    opt::gpu::adam_update(
                        get_gpu_tensor(param_order[i]),
                        get_gpu_tensor(grad_order[i]),
                        get_gpu_tensor(first_moment_order[i]),
                        get_gpu_tensor(second_moment_order[i]),
                        time, alpha, beta1, beta2);
                }
            }
        }

        void const_step_opt::update(std::shared_ptr<vertex> grad)
        {
            tensor_tree::gpu::const_step_update(param, grad, step_size);
        }

        void const_step_momentum_opt::update(std::shared_ptr<vertex> grad)
        {
            tensor_tree::gpu::const_step_update_momentum(param, grad, opt_data, momentum, step_size);
        }

        void adagrad_opt::update(std::shared_ptr<vertex> grad)
        {
            tensor_tree::gpu::adagrad_update(param, grad, accu_grad_sq, step_size);
        }

        void rmsprop_opt::update(std::shared_ptr<vertex> grad)
        {
            tensor_tree::gpu::rmsprop_update(param, grad, accu_grad_sq, decay, step_size);
        }

        void adam_opt::update(std::shared_ptr<vertex> grad)
        {
            tensor_tree::gpu::adam_update(param, grad, first_moment, second_moment,
                time, alpha, beta1, beta2);
        }

    }

}
