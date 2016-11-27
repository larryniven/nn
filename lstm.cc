#include "nn/lstm.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace lstm {

    lstm_step_nn_t make_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input)
    {
        lstm_step_nn_t result;

        result.input = nullptr;

        std::vector<std::shared_ptr<autodiff::op_t>> h_comp { get_var(var_tree->children[2]) };
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp { get_var(var_tree->children[6]) };
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate_comp { get_var(var_tree->children[14]) };

        if (input != nullptr) {
            result.input = autodiff::mul(get_var(var_tree->children[0]), input);
            h_comp.push_back(result.input);
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

    lstm_step_nn_t make_dyer_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input)
    {
        lstm_step_nn_t result;

        autodiff::computation_graph& graph = *tensor_tree::get_var(var_tree->children[0])->graph;
        la::vector<double>& h_vec = autodiff::get_output<la::vector<double>>(
            tensor_tree::get_var(var_tree->children[2]));
        int h_dim = h_vec.size();

        result.input = nullptr;

        std::vector<std::shared_ptr<autodiff::op_t>> h_comp { get_var(var_tree->children[2]) };
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp { get_var(var_tree->children[6]) };

        if (input != nullptr) {
            result.input = autodiff::mul(get_var(var_tree->children[0]), input);
            h_comp.push_back(result.input);
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[3]), input));
        }

        if (output != nullptr) {
            h_comp.push_back(autodiff::mul(get_var(var_tree->children[1]), output));
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[4]), output));
        }

        if (cell != nullptr) {
            input_gate_comp.push_back(autodiff::mul(get_var(var_tree->children[5]), cell));
        }

        la::vector<double> one_vec;
        one_vec.resize(h_dim, 1);
        auto one = graph.var(std::move(one_vec));

        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));
        result.input_gate = autodiff::logistic(autodiff::add(input_gate_comp));
        result.forget_gate = autodiff::sub(one, result.input_gate);

        if (cell != nullptr) {
            result.cell = autodiff::add(
                autodiff::emul(result.forget_gate, cell),
                autodiff::emul(result.input_gate, h));
        } else {
            result.cell = autodiff::emul(result.input_gate, h);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_comp {
            get_var(var_tree->children[10]), autodiff::mul(get_var(var_tree->children[9]), result.cell) };

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

    lstm_builder::~lstm_builder()
    {}

    lstm_nn_t lstm_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
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

    lstm_nn_t dyer_lstm_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        lstm_nn_t result;

        std::shared_ptr<autodiff::op_t> cell = nullptr;
        std::shared_ptr<autodiff::op_t> output = nullptr;

        for (int i = 0; i < feat.size(); ++i) {
            lstm_step_nn_t step_nn = make_dyer_lstm_step_nn(var_tree, cell, output, feat[i]);
            cell = step_nn.cell;
            output = step_nn.output;

            result.cell.push_back(cell);
            result.output.push_back(output);
        }

        return result;
    }

    multilayer_lstm_builder::multilayer_lstm_builder(std::shared_ptr<lstm_builder> builder, int layer)
        : builder(builder), layer(layer)
    {}

    lstm_nn_t multilayer_lstm_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> const *f = &feat;
        lstm_nn_t lstm;
        for (int i = 0; i < layer; ++i) {
             lstm = (*builder)(var_tree->children[i], *f);
             f = &lstm.output;
        }
        return lstm;
    }

    bi_lstm_nn_t make_bi_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        lstm_builder const& builder)
    {
        bi_lstm_nn_t result;

        result.forward_nn = builder(var_tree->children[0], feat);

        std::vector<std::shared_ptr<autodiff::op_t>> rev_feat = feat;
        std::reverse(rev_feat.begin(), rev_feat.end());
        result.backward_nn = builder(var_tree->children[1], rev_feat);
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

    bi_lstm_builder::~bi_lstm_builder()
    {}

    bi_lstm_nn_t bi_lstm_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        return make_bi_lstm_nn(var_tree, feat, lstm_builder{});
    }

    dyer_bi_lstm_builder::dyer_bi_lstm_builder(std::shared_ptr<autodiff::op_t> one)
        : one(one)
    {}

    bi_lstm_nn_t dyer_bi_lstm_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        return make_bi_lstm_nn(var_tree, feat, dyer_lstm_builder{});
    }

    bi_lstm_input_dropout::bi_lstm_input_dropout(
        autodiff::computation_graph& comp_graph,
        int dim,
        std::default_random_engine& gen,
        double prob,
        std::shared_ptr<bi_lstm_builder> builder)
        : comp_graph(comp_graph), dim(dim), gen(gen), prob(prob), builder(builder)
    {}

    bi_lstm_nn_t bi_lstm_input_dropout::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> masked_input;

        std::bernoulli_distribution bernoulli(1.0 - prob);

        for (int j = 0; j < feat.size(); ++j) {
            la::vector<double> v;
            v.resize(dim);
            for (int d = 0; d < v.size(); ++d) {
                v(d) = bernoulli(gen) / (1.0 - prob);
            }
            std::shared_ptr<autodiff::op_t> input_mask = comp_graph.var(std::move(v));
            masked_input.push_back(autodiff::emul(feat[j], input_mask));
        }

        return (*builder)(var_tree, masked_input);
    }

    bi_lstm_input_scaling::bi_lstm_input_scaling(
        autodiff::computation_graph& comp_graph, int dim, double scale,
        std::shared_ptr<bi_lstm_builder> builder)
        : comp_graph(comp_graph), dim(dim), scale(scale), builder(builder)
    {}

    bi_lstm_nn_t bi_lstm_input_scaling::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> masked_input;

        for (int j = 0; j < feat.size(); ++j) {
            la::vector<double> v;
            v.resize(dim, scale);
            std::shared_ptr<autodiff::op_t> input_mask = comp_graph.var(std::move(v));
            masked_input.push_back(autodiff::emul(feat[j], input_mask));
        }

        return (*builder)(var_tree, masked_input);
    }

    bi_lstm_input_subsampling::bi_lstm_input_subsampling(
        std::shared_ptr<bi_lstm_builder> builder)
        : builder(builder), freq(2), once(false)
    {}

    bi_lstm_nn_t bi_lstm_input_subsampling::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        if (once) {
            std::vector<std::shared_ptr<autodiff::op_t>> subsampled_feat = subsample(feat, freq, 0);
            return (*builder)(var_tree, subsampled_feat);
        } else {
            once = true;
            return (*builder)(var_tree, feat);
        }
    }

    stacked_bi_lstm_nn_t make_stacked_bi_lstm_nn(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        bi_lstm_builder const& builder)
    {
        stacked_bi_lstm_nn_t result;

        std::vector<std::shared_ptr<autodiff::op_t>> const* f = &feat;

        for (int i = 0; i < var_tree->children.size(); ++i) {
            result.layer.push_back(builder(var_tree->children[i], *f));
            f = &result.layer.back().output;
        }

        return result;
    }

    std::vector<std::string> subsample(
        std::vector<std::string> const& input,
        int freq, int shift)
    {
        if (freq == 1 && shift == 0) {
            return input;
        }

        std::vector<std::string> result;

        for (int i = 0; i < input.size(); ++i) {
            if ((i - shift) % freq == 0) {
                result.push_back(input[i]);
            }
        }

        return result;
    }

    std::vector<std::shared_ptr<autodiff::op_t>> subsample(
        std::vector<std::shared_ptr<autodiff::op_t>> const& input,
        int freq, int shift)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        // std::cout << "subsample: ";
        for (int i = 0; i < input.size(); ++i) {
            if ((i - shift) % freq == 0) {
                result.push_back(input[i]);
                // std::cout << i << " ";
            }
        }
        // std::cout << std::endl;

        return result;
    }

    std::vector<std::shared_ptr<autodiff::op_t>> upsample(
        std::vector<std::shared_ptr<autodiff::op_t>> const& input,
        int freq, int shift, int length)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        int k = 0;

        // std::cout << "upsample: ";
        for (int i = 0; i < length; ++i) {
            if ((i - shift) % freq == 0) {
                ++k;
            }

            // std::cout << std::max<int>(k - 1, 0) << " ";
            result.push_back(autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                input[std::max<int>(k - 1, 0)] }));
        }
        // std::cout << std::endl;

        return result;
    }

    lstm_nn_t make_zoneout_lstm_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask,
        std::shared_ptr<autodiff::op_t> one)
    {
        lstm_nn_t result;

        std::shared_ptr<autodiff::op_t> cell = nullptr;
        std::shared_ptr<autodiff::op_t> output = nullptr;

        for (int i = 0; i < feat.size(); ++i) {
            lstm_step_nn_t step_nn = make_lstm_step_nn(var_tree, cell, output, feat[i]);

            auto inv_mask = autodiff::sub(one, mask[i]);

            if (cell != nullptr) {
                cell = autodiff::add(autodiff::emul(mask[i], cell),
                    autodiff::emul(inv_mask, step_nn.cell));
            } else {
                cell = step_nn.cell;
            }

            if (output != nullptr) {
                output = autodiff::add(autodiff::emul(mask[i], output),
                    autodiff::emul(inv_mask, step_nn.output));
            } else {
                output = step_nn.output;
            }

            result.cell.push_back(cell);
            result.output.push_back(output);
        }

        return result;
    }

    zoneout_lstm_builder::zoneout_lstm_builder(
            std::vector<std::shared_ptr<autodiff::op_t>> const& mask,
            std::shared_ptr<autodiff::op_t> one)
        : mask(mask), one(one)
    {}

    lstm_nn_t zoneout_lstm_builder::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        return make_zoneout_lstm_nn(var_tree, feat, mask, one);
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

    lstm_step_transcriber::~lstm_step_transcriber()
    {}

    lstm_step_nn_t lstm_step_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input) const
    {
        return make_lstm_step_nn(var_tree, cell, output, input);
    }

    lstm_input_dropout_transcriber::lstm_input_dropout_transcriber(
        std::default_random_engine& gen, double prob,
        std::shared_ptr<lstm_step_transcriber> base)
        : gen(gen), prob(prob), base(base)
    {}

    lstm_step_nn_t lstm_input_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input) const
    {
        la::vector<double> mask;
        auto m_var = tensor_tree::get_var(var_tree->children[0]);
        autodiff::computation_graph& graph = *m_var->graph;
        la::matrix<double>& m = autodiff::get_output<la::matrix<double>>(m_var);
        mask.resize(m.cols());

        std::bernoulli_distribution dist { 1.0 - prob };

        for (int i = 0; i < m.cols(); ++i) {
            mask(i) = dist(gen) / (1.0 - prob);
        }

        return (*base)(var_tree, cell, output, autodiff::emul(graph.var(std::move(mask)), input));
    }

    lstm_output_dropout_transcriber::lstm_output_dropout_transcriber(
        std::default_random_engine& gen, double prob,
        std::shared_ptr<lstm_step_transcriber> base)
        : gen(gen), prob(prob), base(base)
    {}

    lstm_step_nn_t lstm_output_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input) const
    {
        la::vector<double> mask;
        auto m_var = tensor_tree::get_var(var_tree->children[0]);
        autodiff::computation_graph& graph = *m_var->graph;
        la::matrix<double>& m = autodiff::get_output<la::matrix<double>>(m_var);
        mask.resize(m.rows());

        std::bernoulli_distribution dist { 1.0 - prob };

        for (int i = 0; i < m.rows(); ++i) {
            mask(i) = dist(gen) / (1.0 - prob);
        }

        lstm_step_nn_t result = (*base)(var_tree, cell, output, input);

        result.output = autodiff::emul(graph.var(std::move(mask)), result.output);

        return result;
    }

    lstm_step_nn_t dyer_lstm_step_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input) const
    {
        return make_dyer_lstm_step_nn(var_tree, cell, output, input);
    }

    transcriber::~transcriber()
    {}

    lstm_transcriber::lstm_transcriber(
        std::shared_ptr<lstm_step_transcriber> step)
        : step(step)
    {}

    std::vector<std::shared_ptr<autodiff::op_t>>
    lstm_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::shared_ptr<autodiff::op_t> cell = nullptr;
        std::shared_ptr<autodiff::op_t> output = nullptr;

        std::vector<std::shared_ptr<autodiff::op_t>> result;

        for (int i = 0; i < feat.size(); ++i) {
            lstm_step_nn_t step_nn = (*step)(var_tree, cell, output, feat[i]);
            cell = step_nn.cell;
            output = step_nn.output;

            result.push_back(output);
        }

        return result;
    }

    bi_transcriber::bi_transcriber(std::shared_ptr<transcriber> base)
        : base(base)
    {}

    std::vector<std::shared_ptr<autodiff::op_t>> bi_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        std::vector<std::shared_ptr<autodiff::op_t>> f = feat;

        std::vector<std::shared_ptr<autodiff::op_t>> forward;
        std::vector<std::shared_ptr<autodiff::op_t>> backward;

        forward = (*base)(var_tree->children[0], f);

        std::reverse(f.begin(), f.end());

        backward = (*base)(var_tree->children[1], f);

        std::reverse(backward.begin(), backward.end());

        for (int i = 0; i < forward.size(); ++i) {
            result.push_back(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(get_var(var_tree->children[2]), forward[i]),
                autodiff::mul(get_var(var_tree->children[3]), backward[i]),
                get_var(var_tree->children[4])
            }));
        }

        return result;
    }

    std::vector<std::shared_ptr<autodiff::op_t>> layered_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> const *input = &feat;
        std::vector<std::shared_ptr<autodiff::op_t>> output;

        for (int i = 0; i < layer.size(); ++i) {
            output = (*layer[i])(var_tree->children[i], *input);
            input = &output;
        }

        return output;
    }

    subsampled_transcriber::subsampled_transcriber(
        int freq, int shift, std::shared_ptr<transcriber> base)
        : freq(freq), shift(shift), base(base)
    {}

    std::vector<std::shared_ptr<autodiff::op_t>> subsampled_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;
        std::vector<std::shared_ptr<autodiff::op_t>> output = (*base)(var_tree, feat);

        for (int i = 0; i < output.size(); ++i) {
            if ((i - shift) % freq == 0) {
                result.push_back(output[i]);
            }
        }

        return result;
    }

}
