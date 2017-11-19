#ifndef TENSOR_TREE_H
#define TENSOR_TREE_H

#include <vector>
#include <memory>
#include "la/la-cpu.h"
#include "autodiff/autodiff.h"

namespace tensor_tree {

    struct vertex {
        std::string type;

        std::shared_ptr<void> data;

        std::vector<std::shared_ptr<vertex>> children;

        std::string name;
    };

    std::shared_ptr<vertex> make_tensor(std::string name = "");

    template <class vec>
    vec& get_data(std::shared_ptr<vertex> const& t)
    {
        return *std::static_pointer_cast<vec>(t->data);
    }

    la::cpu::tensor<double>& get_tensor(std::shared_ptr<vertex> p);
    std::shared_ptr<autodiff::op_t> get_var(std::shared_ptr<vertex> p);

    std::vector<std::shared_ptr<vertex>> leaves_pre_order(std::shared_ptr<vertex> root);

    void load_tensor(std::shared_ptr<vertex> root, std::istream& is);
    void load_tensor(std::shared_ptr<vertex> root, std::string filename);

    void save_tensor(std::shared_ptr<vertex> root, std::ostream& os);
    void save_tensor(std::shared_ptr<vertex> root, std::string filename);

    void resize_as(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2);

    void axpy(std::shared_ptr<vertex> p1, double d, std::shared_ptr<vertex> p2);
    void zero(std::shared_ptr<vertex> p);

    double norm(std::shared_ptr<vertex> root);

    bool has_nan(std::shared_ptr<vertex> root);

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

    std::shared_ptr<vertex> make_var_tree(autodiff::computation_graph& g,
        std::shared_ptr<vertex> root);

    std::shared_ptr<vertex> copy_tree(std::shared_ptr<vertex> root);

    std::shared_ptr<vertex> shallow_copy(std::shared_ptr<vertex> root);
    std::shared_ptr<vertex> deep_copy(std::shared_ptr<vertex> root);

    void copy_grad(std::shared_ptr<vertex> result, std::shared_ptr<vertex> var_tree);

    void print_tree(std::shared_ptr<vertex> root);

    struct optimizer {

        virtual ~optimizer();

        virtual void update(std::shared_ptr<vertex> grad) = 0;

        virtual void save_opt_data(std::ostream& os) const = 0;

        virtual void load_opt_data(std::istream& is) = 0;

    };

    struct const_step_opt
        : public optimizer {

        std::shared_ptr<vertex> param;
        double step_size;

        const_step_opt(std::shared_ptr<vertex> param,
            double step_size);

        virtual void update(std::shared_ptr<vertex> grad) override;

        virtual void save_opt_data(std::ostream& os) const override;

        virtual void load_opt_data(std::istream& is) override;

    };

    struct const_step_momentum_opt
        : public optimizer {

        std::shared_ptr<vertex> param;
        double step_size;
        double momentum;
        std::shared_ptr<vertex> opt_data;

        const_step_momentum_opt(std::shared_ptr<vertex> param,
            double step_size, double momentum);

        virtual void update(std::shared_ptr<vertex> grad) override;

        virtual void save_opt_data(std::ostream& os) const override;

        virtual void load_opt_data(std::istream& is) override;

    };

    struct adagrad_opt
        : public optimizer {

        std::shared_ptr<vertex> param;
        double step_size;
        std::shared_ptr<vertex> accu_grad_sq;

        adagrad_opt(std::shared_ptr<vertex> param,
            double step_size);

        virtual void update(std::shared_ptr<vertex> grad) override;

        virtual void save_opt_data(std::ostream& os) const override;

        virtual void load_opt_data(std::istream& is) override;

    };

    struct rmsprop_opt
        : public optimizer {

        std::shared_ptr<vertex> param;
        double step_size;
        double decay;
        std::shared_ptr<vertex> accu_grad_sq;

        rmsprop_opt(std::shared_ptr<vertex> param,
            double step_size, double decay);

        virtual void update(std::shared_ptr<vertex> grad) override;

        virtual void save_opt_data(std::ostream& os) const override;

        virtual void load_opt_data(std::istream& is) override;

    };

    struct adam_opt
        : public optimizer {

        std::shared_ptr<vertex> param;
        double alpha;
        double beta1;
        double beta2;

        int time;
        std::shared_ptr<vertex> first_moment;
        std::shared_ptr<vertex> second_moment;

        adam_opt(std::shared_ptr<vertex> param,
            double alpha, double beta1, double beta2);

        virtual void update(std::shared_ptr<vertex> grad) override;

        virtual void save_opt_data(std::ostream& os) const override;

        virtual void load_opt_data(std::istream& is) override;

    };

}

#endif
