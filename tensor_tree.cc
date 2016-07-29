#include "tensor_tree.h"
#include "ebt/ebt.h"
#include "la/la.h"
#include "opt/opt.h"
#include <fstream>

namespace tensor_tree {

    std::shared_ptr<vertex> make_vector(std::string name)
    {
        return std::make_shared<vertex>(vertex {tensor_t::vector, nullptr, {}, name});
    }

    std::shared_ptr<vertex> make_matrix(std::string name)
    {
        return std::make_shared<vertex>(vertex {tensor_t::matrix, nullptr, {}, name});
    }

    la::vector<double>& get_vector(std::shared_ptr<vertex> p)
    {
        if (p->type == tensor_t::vector) {
            return get_data<la::vector<double>>(p);
        } else {
            std::cerr << "expecting tensor_t::vector" << std::endl;
            exit(1);
        }
    }

    la::matrix<double>& get_matrix(std::shared_ptr<vertex> p)
    {
        if (p->type == tensor_t::matrix) {
            return get_data<la::matrix<double>>(p);
        } else {
            std::cerr << "expecting tensor_t::matrix" << std::endl;
            exit(1);
        }
    }

    std::shared_ptr<autodiff::op_t> get_var(std::shared_ptr<vertex> p)
    {
        if (p->type == tensor_t::autodiff_var) {
            return std::static_pointer_cast<autodiff::op_t>(p->data);
        } else {
            std::cerr << "expecting tensor_t::autodiff_var" << std::endl;
            exit(1);
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
            if (v->type == tensor_t::vector) {
                v->data = std::make_shared<la::vector<double>>(
                    ebt::json::load<la::vector<double>>(is));
                std::getline(is, line);
            } else if (v->type == tensor_t::matrix) {
                v->data = std::make_shared<la::matrix<double>>(
                    ebt::json::load<la::matrix<double>>(is));
                std::getline(is, line);
            }
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
            if (v->type == tensor_t::vector) {
                ebt::json::dump(get_data<la::vector<double>>(v), os);
                os << std::endl;
            } else if (v->type == tensor_t::matrix) {
                ebt::json::dump(get_data<la::matrix<double>>(v), os);
                os << std::endl;
            }
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
            if (p1_order[i]->type == tensor_t::vector) {
                la::vector<double> v;
                v.resize(get_data<la::vector<double>>(p2_order[i]).size());
                p1_order[i]->data = std::make_shared<la::vector<double>>(std::move(v));
            } else if (p1_order[i]->type == tensor_t::matrix) {
                la::matrix<double> m;
                auto& m2 = get_data<la::matrix<double>>(p2_order[i]);
                m.resize(m2.rows(), m2.cols());
                p1_order[i]->data = std::make_shared<la::matrix<double>>(std::move(m));
            }
        }
    }

    void imul(std::shared_ptr<vertex> root, double a)
    {
        auto order = leaves_pre_order(root);

        for (auto& t: order) {
            if (t->type == tensor_t::vector) {
                la::imul(get_data<la::vector<double>>(t), a);
            } else if (t->type == tensor_t::matrix) {
                la::imul(get_data<la::matrix<double>>(t), a);
            }
        }
    }

    void iadd(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2)
    {
        auto p1_order = leaves_pre_order(p1);
        auto p2_order = leaves_pre_order(p2);

        for (int i = 0; i < p1_order.size(); ++i) {
            if (p1_order[i]->type == tensor_t::vector) {
                la::iadd(get_data<la::vector<double>>(p1_order[i]),
                    get_data<la::vector<double>>(p2_order[i]));
            } else if (p1_order[i]->type == tensor_t::matrix) {
                la::iadd(get_data<la::matrix<double>>(p1_order[i]),
                    get_data<la::matrix<double>>(p2_order[i]));
            }
        }
    }

    void isub(std::shared_ptr<vertex> p1, std::shared_ptr<vertex> p2)
    {
        auto p1_order = leaves_pre_order(p1);
        auto p2_order = leaves_pre_order(p2);

        for (int i = 0; i < p1_order.size(); ++i) {
            if (p1_order[i]->type == tensor_t::vector) {
                la::isub(get_data<la::vector<double>>(p1_order[i]),
                    get_data<la::vector<double>>(p2_order[i]));
            } else if (p1_order[i]->type == tensor_t::matrix) {
                la::isub(get_data<la::matrix<double>>(p1_order[i]),
                    get_data<la::matrix<double>>(p2_order[i]));
            }
        }
    }

    void zero(std::shared_ptr<vertex> root)
    {
        auto order = leaves_pre_order(root);

        for (auto& t: order) {
            if (t->type == tensor_t::vector) {
                la::zero(get_data<la::vector<double>>(t));
            } else if (t->type == tensor_t::matrix) {
                la::zero(get_data<la::matrix<double>>(t));
            }
        }
    }

    double norm(std::shared_ptr<vertex> root)
    {
        auto order = leaves_pre_order(root);

        double result = 0;

        for (int i = 0; i < order.size(); ++i) {
            if (order[i]->type == tensor_t::vector) {
                result += la::norm(get_vector(order[i]));
            } else if (order[i]->type == tensor_t::matrix) {
                result += la::norm(get_matrix(order[i]));
            }
        }

        return result;
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
            if (grad_order[i]->type == tensor_t::vector) {
                opt::const_step_update_momentum(
                    get_data<la::vector<double>>(param_order[i]),
                    get_data<la::vector<double>>(grad_order[i]),
                    get_data<la::vector<double>>(opt_data_order[i]),
                    momentum, step_size);
            } else if (grad_order[i]->type == tensor_t::matrix) {
                opt::const_step_update_momentum(
                    get_data<la::matrix<double>>(param_order[i]),
                    get_data<la::matrix<double>>(grad_order[i]),
                    get_data<la::matrix<double>>(opt_data_order[i]),
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
            if (grad_order[i]->type == tensor_t::vector) {
                opt::adagrad_update(
                    get_data<la::vector<double>>(param_order[i]),
                    get_data<la::vector<double>>(grad_order[i]),
                    get_data<la::vector<double>>(accu_grad_sq_order[i]),
                    step_size);
            } else if (grad_order[i]->type == tensor_t::matrix) {
                opt::adagrad_update(
                    get_data<la::matrix<double>>(param_order[i]),
                    get_data<la::matrix<double>>(grad_order[i]),
                    get_data<la::matrix<double>>(accu_grad_sq_order[i]),
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
            if (grad_order[i]->type == tensor_t::vector) {
                opt::rmsprop_update(
                    get_data<la::vector<double>>(param_order[i]),
                    get_data<la::vector<double>>(grad_order[i]),
                    get_data<la::vector<double>>(opt_data_order[i]),
                    decay, step_size);
            } else if (grad_order[i]->type == tensor_t::matrix) {
                opt::rmsprop_update(
                    get_data<la::matrix<double>>(param_order[i]),
                    get_data<la::matrix<double>>(grad_order[i]),
                    get_data<la::matrix<double>>(opt_data_order[i]),
                    decay, step_size);
            }
        }
    }

    void adam_update(std::shared_ptr<vertex> param,
        std::shared_ptr<vertex> grad,
        std::shared_ptr<vertex> first_moment,
        std::shared_ptr<vertex> second_moment,
        int time, double alpha, double beta1, double beta2)
    {
        auto param_order = leaves_pre_order(param);
        auto grad_order = leaves_pre_order(grad);
        auto first_moment_order = leaves_pre_order(first_moment);
        auto second_moment_order = leaves_pre_order(second_moment);

        assert(param_order.size() == grad_order.size()
            && grad_order.size() == first_moment_order.size()
            && first_moment_order.size() == second_moment_order.size());

        for (int i = 0; i < param_order.size(); ++i) {
            if (grad_order[i]->type == tensor_t::vector) {
                opt::adam_update(
                    get_data<la::vector<double>>(param_order[i]),
                    get_data<la::vector<double>>(grad_order[i]),
                    get_data<la::vector<double>>(first_moment_order[i]),
                    get_data<la::vector<double>>(second_moment_order[i]),
                    time, alpha, beta1, beta2);
            } else if (grad_order[i]->type == tensor_t::matrix) {
                opt::adam_update(
                    get_data<la::matrix<double>>(param_order[i]),
                    get_data<la::matrix<double>>(grad_order[i]),
                    get_data<la::matrix<double>>(first_moment_order[i]),
                    get_data<la::matrix<double>>(second_moment_order[i]),
                    time, alpha, beta1, beta2);
            }
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
                std::shared_ptr<vertex> k = std::make_shared<vertex>(vertex {tensor_t::nil});

                if (u->type == tensor_t::vector || u->type == tensor_t::matrix) {
                    k->type = tensor_t::autodiff_var;
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
                std::shared_ptr<vertex> k = std::make_shared<vertex>(vertex {tensor_t::nil});

                if (u->type == tensor_t::vector) {
                    k->type = u->type;
                    k->data = std::make_shared<la::vector<double>>(la::vector<double>(get_vector(u)));
                } else if (u->type == tensor_t::matrix) {
                    k->type = u->type;
                    k->data = std::make_shared<la::matrix<double>>(la::matrix<double>(get_matrix(u)));
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
            assert(var_tree_order[i]->type == tensor_t::autodiff_var);

            result_order[i]->data = get_var(var_tree_order[i])->grad;
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

                if (u->type == tensor_t::vector) {
                    std::cout << " type: vector";
                } else if (u->type == tensor_t::matrix) {
                    std::cout << " type: matrix";
                } else if (u->type == tensor_t::autodiff_var) {
                    std::cout << " type: autodiff var";
                    if (get_var(u)->graph != nullptr) {
                        std::cout << " has graph";
                    }
                } else if (u->type == tensor_t::nil) {
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

}
