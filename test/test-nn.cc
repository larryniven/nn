#include <vector>
#include <tuple>
#include <functional>
#include <iostream>
#include "la/la.h"
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"

std::vector<std::tuple<std::string, std::function<void(void)>>> tests = {
    {"test-dyer-lstm", []() {
        la::tensor<double> w_xg {
            {1, 2, 3, 4},
            {2, 2}
        };

        la::tensor<double> w_hg {
            {5, 6, 7, 8},
            {2, 2}
        };

        la::tensor<double> b_g {
            {33, 34},
            {2}
        };

        la::tensor<double> w_xi {
            {9, 10, 11, 12},
            {2, 2}
        };

        la::tensor<double> w_hi {
            {13, 14, 15, 16},
            {2, 2}
        };

        la::tensor<double> w_ci {
            {17, 18, 19, 20},
            {2, 2}
        };

        la::tensor<double> b_i {
            {35, 36},
            {2}
        };

        la::tensor<double> w_xo {
            {21, 22, 23, 24},
            {2, 2}
        };

        la::tensor<double> w_ho {
            {25, 26, 27, 28},
            {2, 2}
        };

        la::tensor<double> w_co {
            {29, 30, 31, 32},
            {2, 2}
        };

        la::tensor<double> b_o {
            {37, 38},
            {2}
        };

        lstm::dyer_lstm_tensor_tree_factory fac;

        std::shared_ptr<tensor_tree::vertex> param = fac();

        param->children[0]->data = std::make_shared<la::tensor<double>>(w_xg);
        param->children[1]->data = std::make_shared<la::tensor<double>>(w_hg);
        param->children[2]->data = std::make_shared<la::tensor<double>>(b_g);

        param->children[3]->data = std::make_shared<la::tensor<double>>(w_xi);
        param->children[4]->data = std::make_shared<la::tensor<double>>(w_hi);
        param->children[5]->data = std::make_shared<la::tensor<double>>(w_ci);
        param->children[6]->data = std::make_shared<la::tensor<double>>(b_i);

        param->children[7]->data = std::make_shared<la::tensor<double>>(w_xo);
        param->children[8]->data = std::make_shared<la::tensor<double>>(w_ho);
        param->children[9]->data = std::make_shared<la::tensor<double>>(w_co);
        param->children[10]->data = std::make_shared<la::tensor<double>>(b_o);

        la::tensor<double> x_vec {
            {1, 2},
            {2}
        };

        la::tensor<double> h_vec {
            {3, 4},
            {2}
        };

        la::tensor<double> cell_vec {
            {5, 6},
            {2}
        };

        autodiff::computation_graph graph;

        auto var_tree = tensor_tree::make_var_tree(graph, param);

        auto output = graph.var(h_vec);
        auto cell = graph.var(cell_vec);
        auto input = graph.var(x_vec);

        lstm::lstm_step_nn_t nn = lstm::make_dyer_lstm_step_nn(var_tree, cell, output, input);

        std::vector<std::shared_ptr<autodiff::op_t>> topo_order;

        for (int i = nn.output->id; i >= 0; --i) {
            topo_order.push_back(graph.vertices.at(i));
            std::cout << graph.vertices.at(i)->name << std::endl;
        }

        la::tensor<double> grad {
            {1, 0},
            {2}
        };

        nn.output->grad = std::make_shared<la::tensor<double>>(grad);

        autodiff::grad(topo_order, autodiff::grad_funcs);
    }},
};

int main()
{
    for (auto& t: tests) {
        std::cout << std::get<0>(t) << std::endl;
        std::get<1>(t)();
    }

    return 0;
}
