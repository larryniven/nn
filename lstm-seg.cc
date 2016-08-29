#include "lstm-seg.h"

namespace lstm_seg {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer,
        std::unordered_map<std::string, std::string> const& args)
    {
        if (ebt::in(std::string("endpoints"), args)) {
            return lstm_seg::endpoints::make_tensor_tree(layer);
        } else {
            return lstm_seg::make_tensor_tree(layer);
        }
    }

    std::shared_ptr<autodiff::op_t> make_pred_nn(
        autodiff::computation_graph& graph,
        lstm::stacked_bi_lstm_nn_t& nn,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<tensor_tree::vertex> param,
        std::unordered_map<std::string, std::string> const& args)
    {
        if (ebt::in(std::string("uniform-att"), args)) {
            la::vector<double>& h = tensor_tree::get_vector(
                param->children[0]->children.back()->children.back());
            return lstm_seg::make_pred_nn_uniform(graph, nn, var_tree, h.size());
        } else if (ebt::in(std::string("endpoints"), args)) {
            return lstm_seg::endpoints::make_pred_nn(graph, nn, var_tree);
        } else {
            return lstm_seg::make_pred_nn(graph, nn, var_tree);
        }
    }

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        tensor_tree::vertex v { tensor_tree::tensor_t::nil };
        v.children.push_back(lstm::make_stacked_bi_lstm_tensor_tree(layer));
        v.children.push_back(tensor_tree::make_matrix());
        v.children.push_back(tensor_tree::make_vector());
        return std::make_shared<tensor_tree::vertex>(v);
    }

    std::shared_ptr<autodiff::op_t> make_pred_nn(
        autodiff::computation_graph& graph,
        lstm::stacked_bi_lstm_nn_t& nn,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        std::shared_ptr<autodiff::op_t> hs = autodiff::col_cat(nn.layer.back().output);
        std::shared_ptr<autodiff::op_t> att_weight
            = autodiff::softmax(autodiff::lmul(tensor_tree::get_var(var_tree->children[2]), hs));

        std::shared_ptr<autodiff::op_t> phi = autodiff::mul(hs, att_weight);

        std::shared_ptr<autodiff::op_t> pred_var = autodiff::logsoftmax(
            autodiff::mul(tensor_tree::get_var(var_tree->children[1]), phi));

        return pred_var;
    }

    std::shared_ptr<autodiff::op_t> make_pred_nn_uniform(
        autodiff::computation_graph& graph,
        lstm::stacked_bi_lstm_nn_t& nn,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        int h_dim)
    {
        std::shared_ptr<autodiff::op_t> h = autodiff::add(nn.layer.back().output);

        la::vector<double> v;
        v.resize(h_dim, 1.0 / nn.layer.back().output.size());
        std::shared_ptr<autodiff::op_t> z = graph.var(v);

        std::shared_ptr<autodiff::op_t> phi = autodiff::emul(h, z);

        std::shared_ptr<autodiff::op_t> pred_var = autodiff::logsoftmax(
            autodiff::mul(tensor_tree::get_var(var_tree->children[1]), phi));

        return pred_var;
    }

    namespace endpoints {

        std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
        {
            tensor_tree::vertex v { tensor_tree::tensor_t::nil };
            v.children.push_back(lstm::make_stacked_bi_lstm_tensor_tree(layer));
            v.children.push_back(tensor_tree::make_matrix());
            v.children.push_back(tensor_tree::make_matrix());
            return std::make_shared<tensor_tree::vertex>(v);
        }

        std::shared_ptr<autodiff::op_t> make_pred_nn(
            autodiff::computation_graph& graph,
            lstm::stacked_bi_lstm_nn_t& nn,
            std::shared_ptr<tensor_tree::vertex> var_tree)
        {
            return autodiff::logsoftmax(autodiff::add(
                autodiff::mul(tensor_tree::get_var(var_tree->children[1]),
                    nn.layer.back().output.front()),
                autodiff::mul(tensor_tree::get_var(var_tree->children[2]),
                    nn.layer.back().output.back())));
        }

    }

    namespace logp {

        std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
        {
            tensor_tree::vertex v { tensor_tree::tensor_t::nil };
            v.children.push_back(lstm::make_stacked_bi_lstm_tensor_tree(layer));
            v.children.push_back(tensor_tree::make_matrix());
            v.children.push_back(tensor_tree::make_vector());
            return std::make_shared<tensor_tree::vertex>(v);
        }

        std::shared_ptr<autodiff::op_t> make_pred_nn(
            autodiff::computation_graph& graph,
            lstm::stacked_bi_lstm_nn_t& nn,
            std::shared_ptr<tensor_tree::vertex> var_tree,
            int label_dim)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> logp;

            for (int i = 0; i < nn.layer.back().output.size(); ++i) {
                logp.push_back(autodiff::logsoftmax(autodiff::add(
                    autodiff::mul(tensor_tree::get_var(var_tree->children[1]), nn.layer.back().output[i]),
                    tensor_tree::get_var(var_tree->children[2]))));
            }

            la::vector<double> v;
            v.resize(label_dim, 1.0 / nn.layer.back().output.size());

            return autodiff::emul(autodiff::add(logp), graph.var(v));
        }

    }

}
