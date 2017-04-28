#include "nn/tensor-tree.h"
#include "ebt/ebt.h"
#include "la/la.h"
#include "opt/opt.h"
#include <fstream>
#include <exception>

using namespace std::string_literals;

namespace tensor_tree {

    std::shared_ptr<vertex> make_tensor(std::string name)
    {
        return std::make_shared<vertex>(vertex {"tensor"s, nullptr, {}, name});
    }

    la::tensor<double>& get_tensor(std::shared_ptr<vertex> p)
    {
        if (p->type == "tensor") {
            return get_data<la::tensor<double>>(p);
        } else {
            throw std::logic_error("expecting tensor");
        }
    }

    std::shared_ptr<autodiff::op_t> get_var(std::shared_ptr<vertex> p)
    {
        if (p->type == "autodiff-var") {
            return std::static_pointer_cast<autodiff::op_t>(p->data);
        } else {
            throw std::logic_error("expecting autodiff-var");
        }
    }

    std::vector<std::shared_ptr<vertex>> leaves_pre_order(std::shared_ptr<vertex> root)
    {
        std::vector<std::shared_ptr<vertex>> result;

        std::vector<std::shared_ptr<vertex>> stack;

        stack.push_back(root);

        while (stack.size() > 0) {
            std::shared_ptr<vertex> u = stack.back();
            stack.pop_back();

            if (u->children.size() == 0) {
                result.push_back(u);
            } else {
                for (int i = u->children.size() - 1; i >= 0; --i) {
                    stack.push_back(u->children[i]);
                }
            }
        }

        return result;
    }

    void load_tensor(std::shared_ptr<vertex> root, std::istream& is)
    {
        std::string line;
        std::vector<std::shared_ptr<vertex>> order = leaves_pre_order(root);

        for (auto& v: order) {
            v->data = std::make_shared<la::tensor<double>>(
                ebt::json::load<la::tensor<double>>(is));
            std::getline(is, line);
        }
    }

    void load_tensor(std::shared_ptr<vertex> root, std::string filename)
    {
        std::ifstream ifs { filename };

        return load_tensor(root, ifs);
    }

    void save_tensor(std::shared_ptr<vertex> root, std::ostream& os)
    {
        std::vector<std::shared_ptr<vertex>> order = leaves_pre_order(root);

        for (auto& v: order) {
            ebt::json::dump(get_data<la::tensor<double>>(v), os);
            os << std::endl;
        }
    }

    void save_tensor(std::shared_ptr<vertex> root, std::string filename)
    {
        std::ofstream ofs { filename };

        save_tensor(root, ofs);
    }

    void resize_as(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2)
    {
        auto p1_order = leaves_pre_order(p1);
        auto p2_order = leaves_pre_order(p2);

        for (int i = 0; i < p1_order.size(); ++i) {
            la::tensor<double> m;
            auto& m2 = get_data<la::tensor<double>>(p2_order[i]);
            la::resize_as(m, m2);
            p1_order[i]->data = std::make_shared<la::tensor<double>>(std::move(m));
        }
    }

    void imul(std::shared_ptr<vertex> root, double a)
    {
        auto order = leaves_pre_order(root);

        for (auto& t: order) {
            if (t->data == nullptr) {
                continue;
            }

            la::imul(get_data<la::tensor<double>>(t), a);
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

            la::iadd(get_data<la::tensor<double>>(p1_order[1]),
                get_data<la::tensor<double>>(p2_order[i]));
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

            la::isub(get_data<la::tensor<double>>(p1_order[i]),
                get_data<la::tensor<double>>(p2_order[i]));
        }
    }

    void zero(std::shared_ptr<vertex> root)
    {
        auto order = leaves_pre_order(root);

        for (auto& t: order) {
            if (t->data == nullptr) {
                continue;
            }

            la::zero(get_data<la::tensor<double>>(t));
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

            result += std::pow(la::norm(get_tensor(order[i])), 2);
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

            if (la::has_nan(get_tensor(order[i]))) {
                return true;
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

            opt::const_step_update(
                get_data<la::tensor<double>>(param_order[i]),
                get_data<la::tensor<double>>(grad_order[i]),
                step_size);
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

            opt::const_step_update_momentum(
                get_data<la::tensor<double>>(param_order[i]),
                get_data<la::tensor<double>>(grad_order[i]),
                get_data<la::tensor<double>>(opt_data_order[i]),
                momentum, step_size);
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

            opt::adagrad_update(
                get_data<la::tensor<double>>(param_order[i]),
                get_data<la::tensor<double>>(grad_order[i]),
                get_data<la::tensor<double>>(accu_grad_sq_order[i]),
                step_size);
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

            opt::rmsprop_update(
                get_data<la::tensor<double>>(param_order[i]),
                get_data<la::tensor<double>>(grad_order[i]),
                get_data<la::tensor<double>>(opt_data_order[i]),
                decay, step_size);
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

            opt::adam_update(
                get_data<la::tensor<double>>(param_order[i]),
                get_data<la::tensor<double>>(grad_order[i]),
                get_data<la::tensor<double>>(first_moment_order[i]),
                get_data<la::tensor<double>>(second_moment_order[i]),
                time, alpha, beta1, beta2);
        }
    }

    std::shared_ptr<vertex> make_var_tree(autodiff::computation_graph& g,
        std::shared_ptr<vertex> root)
    {
        std::unordered_map<std::shared_ptr<vertex>, std::shared_ptr<vertex>> vertex_map;

        std::vector<std::tuple<std::shared_ptr<vertex>, bool>> stack;

        stack.push_back(std::make_tuple(root, false));

        while (stack.size() > 0) {
            std::shared_ptr<vertex> u;
            bool finished;
            std::tie(u, finished) = stack.back();
            stack.pop_back();

            if (!finished) {
                stack.push_back(std::make_tuple(u, true));

                for (int i = u->children.size() - 1; i >= 0; --i) {
                    stack.push_back(std::make_tuple(u->children[i], false));
                }
            } else {
                std::shared_ptr<vertex> k = std::make_shared<vertex>(vertex {"nil"});

                if (u->type != "nil") {
                    k->type = "autodiff-var";
                    auto v = g.var();
                    v->grad_needed = true;
                    v->output = u->data;
                    k->data = v;
                }

                vertex_map[u] = k;

                for (int i = 0; i < u->children.size(); ++i) {
                    k->children.push_back(vertex_map.at(u->children[i]));
                }
            }
        }

        return vertex_map.at(root);
    }

    std::shared_ptr<vertex> copy_tree(std::shared_ptr<vertex> root)
    {
        return deep_copy(root);
    }

    std::shared_ptr<vertex> shallow_copy(std::shared_ptr<vertex> root)
    {
        std::unordered_map<std::shared_ptr<vertex>, std::shared_ptr<vertex>> vertex_map;

        std::vector<std::tuple<std::shared_ptr<vertex>, bool>> stack;

        stack.push_back(std::make_tuple(root, false));

        while (stack.size() > 0) {
            std::shared_ptr<vertex> u;
            bool finished;
            std::tie(u, finished) = stack.back();
            stack.pop_back();

            if (!finished) {
                stack.push_back(std::make_tuple(u, true));

                for (int i = u->children.size() - 1; i >= 0; --i) {
                    stack.push_back(std::make_tuple(u->children[i], false));
                }
            } else {
                std::shared_ptr<vertex> k = std::make_shared<vertex>(vertex {"nil"});

                if (u->type != "nil") {
                    k->type = u->type;
                }

                vertex_map[u] = k;

                for (int i = 0; i < u->children.size(); ++i) {
                    k->children.push_back(vertex_map.at(u->children[i]));
                }
            }
        }

        return vertex_map.at(root);
    }

    std::shared_ptr<vertex> deep_copy(std::shared_ptr<vertex> root)
    {
        std::unordered_map<std::shared_ptr<vertex>, std::shared_ptr<vertex>> vertex_map;

        std::vector<std::tuple<std::shared_ptr<vertex>, bool>> stack;

        stack.push_back(std::make_tuple(root, false));

        while (stack.size() > 0) {
            std::shared_ptr<vertex> u;
            bool finished;
            std::tie(u, finished) = stack.back();
            stack.pop_back();

            if (!finished) {
                stack.push_back(std::make_tuple(u, true));

                for (int i = u->children.size() - 1; i >= 0; --i) {
                    stack.push_back(std::make_tuple(u->children[i], false));
                }
            } else {
                std::shared_ptr<vertex> k = std::make_shared<vertex>(vertex {"nil"});

                if (u->type != "nil") {
                    k->name = u->name;
                    k->type = u->type;
                    k->data = std::make_shared<la::tensor<double>>(la::tensor<double>(get_tensor(u)));
                }

                vertex_map[u] = k;

                for (int i = 0; i < u->children.size(); ++i) {
                    k->children.push_back(vertex_map.at(u->children[i]));
                }
            }
        }

        return vertex_map.at(root);
    }

    void copy_grad(std::shared_ptr<vertex> result, std::shared_ptr<vertex> var_tree)
    {
        auto result_order = leaves_pre_order(result);
        auto var_tree_order = leaves_pre_order(var_tree);

        assert(result_order.size() == var_tree_order.size());

        for (int i = 0; i < result_order.size(); ++i) {
            assert(var_tree_order[i]->type == "autodiff-var");

            auto grad = get_var(var_tree_order[i])->grad;

            if (grad != nullptr) {
                result_order[i]->data = grad;
            }
        }
    }

    void print_tree(std::shared_ptr<vertex> root)
    {
        std::vector<std::tuple<std::shared_ptr<vertex>, bool>> stack;

        std::vector<std::shared_ptr<vertex>> path;

        stack.push_back(std::make_tuple(root, false));

        while (stack.size() > 0) {
            std::shared_ptr<vertex> u;
            bool finished;
            std::tie(u, finished) = stack.back();
            stack.pop_back();

            if (!finished) {
                path.push_back(u);

                stack.push_back(std::make_tuple(u, true));

                for (int i = u->children.size() - 1; i >= 0; --i) {
                    stack.push_back(std::make_tuple(u->children[i], false));
                }

                for (int i = 0; i < path.size() - 1; ++i) {
                    std::cout << "  ";
                }

                std::cout << "name: " << u->name;

                if (u->type == "tensor") {
                    std::cout << " type: tensor";
                } else if (u->type == "autodiff-var") {
                    std::cout << " type: autodiff var";
                    if (get_var(u)->graph != nullptr) {
                        std::cout << " has graph";
                    }
                } else if (u->type == "nil") {
                    std::cout << " type: nil";
                }

                if (u->data != nullptr) {
                    std::cout << " has data";
                } else {
                    std::cout << " no data";
                }

                std::cout << std::endl;
            } else {
                path.pop_back();
            }
        }
    }

    optimizer::~optimizer()
    {}

    const_step_opt::const_step_opt(std::shared_ptr<vertex> param,
        double step_size)
        : param(param), step_size(step_size)
    {}

    void const_step_opt::update(std::shared_ptr<vertex> grad)
    {
        const_step_update(param, grad, step_size);
    }

    void const_step_opt::save_opt_data(std::ostream& os) const
    {
    }

    void const_step_opt::load_opt_data(std::istream& is)
    {
    }

    const_step_momentum_opt::const_step_momentum_opt(std::shared_ptr<vertex> param,
        double step_size, double momentum)
        : param(param), step_size(step_size), momentum(momentum)
    {
        opt_data = shallow_copy(param);
    }

    void const_step_momentum_opt::update(std::shared_ptr<vertex> grad)
    {
        const_step_update_momentum(param, grad, opt_data, momentum, step_size);
    }

    void const_step_momentum_opt::save_opt_data(std::ostream& os) const
    {
        save_tensor(opt_data, os);
    }

    void const_step_momentum_opt::load_opt_data(std::istream& is)
    {
        load_tensor(opt_data, is);
    }

    adagrad_opt::adagrad_opt(std::shared_ptr<vertex> param,
        double step_size)
        : param(param), step_size(step_size)
    {
        accu_grad_sq = shallow_copy(param);
    }

    void adagrad_opt::update(std::shared_ptr<vertex> grad)
    {
        adagrad_update(param, grad, accu_grad_sq, step_size);
    }

    void adagrad_opt::save_opt_data(std::ostream& os) const
    {
        save_tensor(accu_grad_sq, os);
    }

    void adagrad_opt::load_opt_data(std::istream& is)
    {
        load_tensor(accu_grad_sq, is);
    }

    rmsprop_opt::rmsprop_opt(std::shared_ptr<vertex> param,
        double step_size, double decay)
        : param(param), step_size(step_size), decay(decay)
    {
        accu_grad_sq = shallow_copy(param);
    }

    void rmsprop_opt::update(std::shared_ptr<vertex> grad)
    {
        rmsprop_update(param, grad, accu_grad_sq, decay, step_size);
    }

    void rmsprop_opt::save_opt_data(std::ostream& os) const
    {
        save_tensor(accu_grad_sq, os);
    }

    void rmsprop_opt::load_opt_data(std::istream& is)
    {
        load_tensor(accu_grad_sq, is);
    }

    adam_opt::adam_opt(std::shared_ptr<vertex> param,
        double alpha, double beta1, double beta2)
        : param(param)
        , alpha(alpha), beta1(beta1), beta2(beta2)
    {
        first_moment = shallow_copy(param);
        second_moment = shallow_copy(param);
    }

    void adam_opt::update(std::shared_ptr<vertex> grad)
    {
        adam_update(param, grad, first_moment, second_moment,
            time, alpha, beta1, beta2);
    }

    void adam_opt::save_opt_data(std::ostream& os) const
    {
        os << time << std::endl;
        save_tensor(first_moment, os);
        save_tensor(second_moment, os);
    }

    void adam_opt::load_opt_data(std::istream& is)
    {
        std::string line;
        std::getline(is, line);
        time = std::stoi(line);
        load_tensor(first_moment, is);
        load_tensor(second_moment, is);
    }

}
