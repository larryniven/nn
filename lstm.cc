#include "nn/lstm.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace lstm {

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree()
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

        // 0
        root.children.push_back(tensor_tree::make_matrix("input -> hidden"));
        root.children.push_back(tensor_tree::make_matrix("output -> hidden"));
        root.children.push_back(tensor_tree::make_vector("hidden bias"));

        // 3
        root.children.push_back(tensor_tree::make_matrix("input -> input gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> input gate"));
        root.children.push_back(tensor_tree::make_vector("input gate peep"));
        root.children.push_back(tensor_tree::make_vector("input gate bias"));

        // 7
        root.children.push_back(tensor_tree::make_matrix("input -> output gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> output gate"));
        root.children.push_back(tensor_tree::make_vector("output gate peep"));
        root.children.push_back(tensor_tree::make_vector("output gate bias"));

        // 11
        root.children.push_back(tensor_tree::make_matrix("input -> forget gate"));
        root.children.push_back(tensor_tree::make_matrix("output -> forget gate"));
        root.children.push_back(tensor_tree::make_vector("forget gate peep"));
        root.children.push_back(tensor_tree::make_vector("forget gate bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    lstm_step_nn_t make_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input)
    {
        lstm_step_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> h_comp { get_var(var_tree->children[2]) };
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp { get_var(var_tree->children[6]) };
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate_comp { get_var(var_tree->children[14]) };

        if (input != nullptr) {
            h_comp.push_back(autodiff::mul(get_var(var_tree->children[0]), input));
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[3]), input));
            forget_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[11]), input));
        }

        if (output != nullptr) {
            h_comp.push_back(autodiff::mul(get_var(var_tree->children[1]), output));
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[4]), output));
            forget_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[12]), output));
        }

        if (cell != nullptr) {
            input_gate_comp.push_back(autodiff::emul(get_var(var_tree->children[5]), cell));
            forget_gate_comp.push_back(autodiff::emul(get_var(var_tree->children[13]), cell));
        }

        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));
        result.input_gate = autodiff::logistic(autodiff::add(input_gate_comp));
        result.forget_gate = autodiff::logistic(autodiff::add(forget_gate_comp));

        if (cell != nullptr) {
            result.cell = autodiff::add(
                autodiff::emul(result.forget_gate, cell),
                autodiff::emul(result.input_gate, h));
        } else {
            result.cell = autodiff::emul(result.input_gate, h);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_comp {
            get_var(var_tree->children[10]), autodiff::emul(get_var(var_tree->children[9]), result.cell) };

        if (input != nullptr) {
            output_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[7]), input));
        }

        if (output != nullptr) {
            output_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[8]), output));
        }

        result.output_gate = autodiff::logistic(autodiff::add(output_gate_comp));

        result.output = autodiff::emul(result.output_gate,
            autodiff::tanh(result.cell));

        return result;
    }

    lstm_nn_t make_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat)
    {
        lstm_nn_t result;

        std::shared_ptr<autodiff::op_t> cell = nullptr;
        std::shared_ptr<autodiff::op_t> output = nullptr;

        for (int i = 0; i < feat.size(); ++i) {
            lstm_step_nn_t step_nn = make_lstm_step_nn(var_tree, cell, output, feat[i]);
            cell = step_nn.cell;
            output = step_nn.output;

            result.cell.push_back(cell);
            result.output.push_back(output);
        }

        return result;
    }

    std::shared_ptr<tensor_tree::vertex> make_bi_lstm_tensor_tree()
    {
        tensor_tree::vertex root;

        root.children.push_back(make_lstm_tensor_tree());
        root.children.push_back(make_lstm_tensor_tree());

        root.children.push_back(tensor_tree::make_matrix("forward output weight"));
        root.children.push_back(tensor_tree::make_matrix("backward output weight"));
        root.children.push_back(tensor_tree::make_vector("output bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    bi_lstm_nn_t make_bi_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat)
    {
        bi_lstm_nn_t result;

        result.forward_nn = make_lstm_nn(var_tree->children[0], feat);

        std::vector<std::shared_ptr<autodiff::op_t>> rev_feat = feat;
        std::reverse(rev_feat.begin(), rev_feat.end());
        result.backward_nn = make_lstm_nn(var_tree->children[1], rev_feat);
        std::reverse(result.backward_nn.cell.begin(), result.backward_nn.cell.end());
        std::reverse(result.backward_nn.output.begin(), result.backward_nn.output.end());

        for (int i = 0; i < result.forward_nn.output.size(); ++i) {
            result.output.push_back(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(get_var(var_tree->children[2]), result.forward_nn.output[i]),
                autodiff::mul(get_var(var_tree->children[3]), result.backward_nn.output[i]),
                get_var(var_tree->children[4])
            }));
        }

        return result;
    }

    std::shared_ptr<tensor_tree::vertex> make_stacked_bi_lstm_tensor_tree(int layer)
    {
        tensor_tree::vertex root;

        for (int i = 0; i < layer; ++i) {
            root.children.push_back(make_bi_lstm_tensor_tree());
        }

        return std::make_shared<tensor_tree::vertex>(root);
    }

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat)
    {
        stacked_bi_lstm_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> const* f = &feat;

        for (int i = 0; i < var_tree->children.size(); ++i) {
            result.layer.push_back(make_bi_lstm_nn(var_tree->children[i], *f));
            f = &result.layer.back().output;
        }

        return result;
    }

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn_with_dropout(
        autodiff::computation_graph& g,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::default_random_engine& gen, double prob)
    {
        stacked_bi_lstm_nn_t result;

        std::bernoulli_distribution bernoulli { 1 - prob };

        std::vector<std::shared_ptr<autodiff::op_t>> const* f = &feat;

        for (int i = 0; i < var_tree->children.size(); ++i) {
            std::vector<std::shared_ptr<autodiff::op_t>> masked_input;

            auto& m = autodiff::get_output<la::matrix<double>>(
                get_var(var_tree->children[i]->children[0]->children[0]));

            for (int j = 0; j < f->size(); ++j) {
                la::vector<double> v;
                v.resize(m.cols());
                for (int d = 0; d < v.size(); ++d) {
                    v(d) = bernoulli(gen);
                }
                std::shared_ptr<autodiff::op_t> input_mask = g.var(std::move(v));
                masked_input.push_back(autodiff::emul((*f)[j], input_mask));
            }

            result.layer.push_back(make_bi_lstm_nn(var_tree->children[i], masked_input));

            f = &result.layer.back().output;
        }

        auto& v = autodiff::get_output<la::vector<double>>(
            get_var(var_tree->children.back()->children.back()));

        for (int i = 0; i < result.layer.back().output.size(); ++i) {
            la::vector<double> u;
            u.resize(v.size());

            for (int j = 0; j < v.size(); ++j) {
                u(j) = bernoulli(gen);
            }

            std::shared_ptr<autodiff::op_t> output_mask = g.var(std::move(u));

            result.layer.back().output[i] = autodiff::emul(result.layer.back().output[i], output_mask);
        }

        return result;
    }

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn_with_dropout(
        autodiff::computation_graph& g,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        double prob)
    {
        stacked_bi_lstm_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> const* f = &feat;

        for (int i = 0; i < var_tree->children.size(); ++i) {
            std::vector<std::shared_ptr<autodiff::op_t>> masked_input;

            auto& m = autodiff::get_output<la::matrix<double>>(
                get_var(var_tree->children[i]->children[0]->children[0]));

            for (int j = 0; j < f->size(); ++j) {
                la::vector<double> v;
                v.resize(m.cols());
                for (int d = 0; d < v.size(); ++d) {
                    v(d) = 1 - prob;
                }
                std::shared_ptr<autodiff::op_t> input_mask = g.var(std::move(v));
                masked_input.push_back(autodiff::emul((*f)[j], input_mask));
            }

            result.layer.push_back(make_bi_lstm_nn(var_tree->children[i], masked_input));

            f = &result.layer.back().output;
        }

        auto& v = autodiff::get_output<la::vector<double>>(
            get_var(var_tree->children.back()->children.back()));

        for (int i = 0; i < result.layer.back().output.size(); ++i) {
            la::vector<double> u;
            u.resize(v.size());

            for (int j = 0; j < v.size(); ++j) {
                u(j) = 1 - prob;
            }

            std::shared_ptr<autodiff::op_t> output_mask = g.var(std::move(u));

            result.layer.back().output[i] = autodiff::emul(result.layer.back().output[i], output_mask);
        }

        return result;
    }

#if 0
    lstm2d_nn_t make_lstm2d_nn(autodiff::computation_graph& graph,
        lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        lstm2d_nn_t result;

        result.h_nn = make_lstm_unit_nn(graph, param.h_param);
        result.v_nn = make_lstm_unit_nn(graph, param.v_param);

        result.output_h_weight = graph.var(param.output_h_weight);
        result.output_v_weight = graph.var(param.output_v_weight);
        result.output_bias = graph.var(param.output_bias);

        std::shared_ptr<autodiff::op_t> h_cell = nullptr;
        std::shared_ptr<autodiff::op_t> h_output = nullptr;

        for (int i = 0; i < inputs.size(); ++i) {
            lstm_step_nn_t h_step = make_lstm_step(result.h_nn, h_cell, h_output, inputs[i]);
            h_cell = h_step.cell;
            h_output = h_step.output;

            result.h_cell.push_back(h_step.cell);
            result.h_output.push_back(h_step.output);

            lstm_step_nn_t v_step = make_lstm_step(result.v_nn, inputs[i], inputs[i], h_output);

            result.v_cell.push_back(v_step.cell);
            result.v_output.push_back(v_step.output);
        }

        for (int i = 0; i < result.h_cell.size(); ++i) {
            result.output.push_back(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.output_h_weight, result.h_output[i]),
                autodiff::mul(result.output_v_weight, result.v_output[i]),
                result.output_bias}));
        }

        return result;
    }

    lstm2d_nn_t stack_lstm2d(autodiff::computation_graph& graph,
        lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs,
        std::vector<std::shared_ptr<autodiff::op_t>> const& v_output,
        std::vector<std::shared_ptr<autodiff::op_t>> const& v_cell)
    {
        lstm2d_nn_t result;

        result.h_nn = make_lstm_unit_nn(graph, param.h_param);
        result.v_nn = make_lstm_unit_nn(graph, param.v_param);

        result.output_h_weight = graph.var(param.output_h_weight);
        result.output_v_weight = graph.var(param.output_v_weight);
        result.output_bias = graph.var(param.output_bias);

        std::shared_ptr<autodiff::op_t> h_cell = nullptr;
        std::shared_ptr<autodiff::op_t> h_output = nullptr;

        for (int i = 0; i < inputs.size(); ++i) {
            lstm_step_nn_t h_step = make_lstm_step(result.h_nn, h_cell, h_output, inputs[i]);
            h_cell = h_step.cell;
            h_output = h_step.output;

            result.h_cell.push_back(h_step.cell);
            result.h_output.push_back(h_step.output);

            lstm_step_nn_t v_step = make_lstm_step(result.v_nn, v_cell[i], v_output[i], h_output);

            result.v_cell.push_back(v_step.cell);
            result.v_output.push_back(v_step.output);
        }

        for (int i = 0; i < result.h_cell.size(); ++i) {
            result.output.push_back(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.output_h_weight, result.h_output[i]),
                autodiff::mul(result.output_v_weight, result.v_output[i]),
                result.output_bias}));
        }

        return result;
    }

    bi_lstm2d_nn_t stack_bi_lstm2d(autodiff::computation_graph& graph,
        bi_lstm2d_param_t const& param, bi_lstm2d_nn_t const& prev)
    {
        bi_lstm2d_nn_t result;

        result.forward_nn = stack_lstm2d(graph, param.forward_param, prev.output,
            prev.forward_nn.v_output, prev.forward_nn.v_cell);

        std::vector<std::shared_ptr<autodiff::op_t>> rev_output = prev.output;
        std::reverse(rev_output.begin(), rev_output.end());

        result.backward_nn = stack_lstm2d(graph, param.backward_param, rev_output,
            prev.backward_nn.v_output, prev.backward_nn.v_cell);

        std::reverse(result.backward_nn.h_cell.begin(), result.backward_nn.h_cell.end());
        std::reverse(result.backward_nn.h_output.begin(), result.backward_nn.h_output.end());
        std::reverse(result.backward_nn.v_cell.begin(), result.backward_nn.v_cell.end());
        std::reverse(result.backward_nn.v_output.begin(), result.backward_nn.v_output.end());
        std::reverse(result.backward_nn.output.begin(), result.backward_nn.output.end());

        for (int i = 0; i < result.forward_nn.output.size(); ++i) {
            result.output.push_back(autodiff::add(
                result.forward_nn.output[i], result.backward_nn.output[i]));
        }

        return result;
    }

    db_lstm2d_nn_t make_db_lstm2d_nn(autodiff::computation_graph& graph,
        db_lstm2d_param_t const& param,
        std::vector<std::shared_ptr<autodiff::op_t>> const& inputs)
    {
        db_lstm2d_nn_t result;

        result.layer.push_back(make_bi_lstm2d_nn(graph, param.layer[0], inputs));

        for (int i = 1; i < param.layer.size(); ++i) {
            result.layer.push_back(stack_bi_lstm2d(graph, param.layer[i], result.layer.back()));
        }

        return result;
    }

#endif

}
