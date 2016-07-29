#ifndef TENSOR_TREE_H
#define TENSOR_TREE_H

#include <vector>
#include <memory>
#include "la/la.h"
#include "autodiff/autodiff.h"

namespace tensor_tree {

    enum class tensor_t {
        nil,
        vector,
        matrix,
        autodiff_var
    };

    struct vertex {
        tensor_t type;

        std::shared_ptr<void> data;

        std::vector<std::shared_ptr<vertex>> children;

        std::string name;
    };

    std::shared_ptr<vertex> make_vector(std::string name = "");
    std::shared_ptr<vertex> make_matrix(std::string name = "");

    template <class vec>
    vec& get_data(std::shared_ptr<vertex> const& t)
    {
        return *std::static_pointer_cast<vec>(t->data);
    }

    la::vector<double>& get_vector(std::shared_ptr<vertex> p);
    la::matrix<double>& get_matrix(std::shared_ptr<vertex> p);
    std::shared_ptr<autodiff::op_t> get_var(std::shared_ptr<vertex> p);

    std::vector<std::shared_ptr<vertex>> pre_order(std::shared_ptr<vertex> root);

    void load_tensor(std::shared_ptr<vertex> root, std::istream& is);
    void load_tensor(std::shared_ptr<vertex> root, std::string filename);

    void save_tensor(std::shared_ptr<vertex> root, std::ostream& os);
    void save_tensor(std::shared_ptr<vertex> root, std::string filename);

    void resize_as(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2);

    void imul(std::shared_ptr<vertex> root, double a);
    void iadd(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2);
    void isub(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2);
    void zero(std::shared_ptr<vertex> p);

    double norm(std::shared_ptr<vertex> root);

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
        int time, double alpha, double beta1, double beta2);

    std::shared_ptr<vertex> make_var_tree(autodiff::computation_graph& g,
        std::shared_ptr<vertex> root);

    std::shared_ptr<vertex> copy_tree(std::shared_ptr<vertex> root);

    void copy_grad(std::shared_ptr<vertex> result, std::shared_ptr<vertex> var_tree);

    void print_tree(std::shared_ptr<vertex> root);
}

#endif
