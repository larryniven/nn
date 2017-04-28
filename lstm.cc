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
            result.input = autodiff::mul(input, get_var(var_tree->children[0]));
            h_comp.push_back(result.input);
            input_gate_comp.push_back(autodiff::mul(input, get_var(var_tree->children[3])));
            forget_gate_comp.push_back(autodiff::mul(input, get_var(var_tree->children[11])));
        }

        if (output != nullptr) {
            h_comp.push_back(autodiff::mul(output, get_var(var_tree->children[1])));
            input_gate_comp.push_back(autodiff::mul(output, get_var(var_tree->children[4])));
            forget_gate_comp.push_back(autodiff::mul(output, get_var(var_tree->children[12])));
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
            output_gate_comp.push_back(autodiff::mul(input, get_var(var_tree->children[7])));
        }

        if (output != nullptr) {
            output_gate_comp.push_back(autodiff::mul(output, get_var(var_tree->children[8])));
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

        result.input = nullptr;

        std::vector<std::shared_ptr<autodiff::op_t>> h_comp;
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp;

        if (input != nullptr) {
            result.input = autodiff::mul(input, get_var(var_tree->children[0]));
            h_comp.push_back(result.input);
            input_gate_comp.push_back(autodiff::mul(input, get_var(var_tree->children[3])));
        }

        if (output != nullptr) {
            h_comp.push_back(autodiff::mul(output, get_var(var_tree->children[1])));
            input_gate_comp.push_back(autodiff::mul(output, get_var(var_tree->children[4])));
        }

        if (cell != nullptr) {
            input_gate_comp.push_back(autodiff::mul(cell, get_var(var_tree->children[5])));
        }

        assert(h_comp.size() > 0);
        h_comp.push_back(autodiff::rep_row_to(get_var(var_tree->children[2]), h_comp.back()));
        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));

        assert(input_gate_comp.size() > 0);
        input_gate_comp.push_back(autodiff::rep_row_to(get_var(var_tree->children[6]),
            input_gate_comp.back()));
        result.input_gate = autodiff::logistic(autodiff::add(input_gate_comp));

        if (cell != nullptr) {
            auto one = autodiff::resize_as(result.input_gate, 1);

            result.forget_gate = autodiff::sub(one, result.input_gate);

            result.cell = autodiff::add(
                autodiff::emul(result.forget_gate, cell),
                autodiff::emul(result.input_gate, h));
        } else {
            result.cell = autodiff::emul(result.input_gate, h);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_comp {
            autodiff::mul(result.cell, get_var(var_tree->children[9])) };

        output_gate_comp.push_back(autodiff::rep_row_to(get_var(var_tree->children[10]),
            output_gate_comp.back()));

        if (input != nullptr) {
            output_gate_comp.push_back(autodiff::mul(input, get_var(var_tree->children[7])));
        }

        if (output != nullptr) {
            output_gate_comp.push_back(autodiff::mul(output, get_var(var_tree->children[8])));
        }

        result.output_gate = autodiff::logistic(autodiff::add(output_gate_comp));

        result.output = autodiff::emul(result.output_gate,
            autodiff::tanh(result.cell));

        return result;
    }

    step_transcriber::~step_transcriber()
    {}

    lstm_step_transcriber::lstm_step_transcriber()
        : cell(nullptr), output(nullptr)
    {}

    std::shared_ptr<autodiff::op_t> lstm_step_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        lstm_step_nn_t nn = make_lstm_step_nn(var_tree, cell, output, input);

        cell = nn.cell;
        output = nn.output;

        return output;
    }

    input_dropout_transcriber::input_dropout_transcriber(
        std::default_random_engine& gen, double prob,
        std::shared_ptr<step_transcriber> base)
        : gen(gen), prob(prob), base(base)
    {}

    std::shared_ptr<autodiff::op_t> input_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        la::tensor<double> mask;
        auto m_var = tensor_tree::get_var(var_tree->children[0]);
        autodiff::computation_graph& graph = *m_var->graph;
        la::tensor_like<double>& m = autodiff::get_output<la::tensor_like<double>>(m_var);
        mask.resize({m.size(0)});

        std::bernoulli_distribution dist { 1.0 - prob };

        for (int i = 0; i < m.size(0); ++i) {
            mask({i}) = dist(gen) / (1.0 - prob);
        }

        auto mask_var = graph.var(std::move(mask));
        return (*base)(var_tree, autodiff::emul(mask_var, input));
    }

    output_dropout_transcriber::output_dropout_transcriber(
        std::default_random_engine& gen, double prob,
        std::shared_ptr<step_transcriber> base)
        : gen(gen), prob(prob), base(base)
    {}

    std::shared_ptr<autodiff::op_t> output_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        la::tensor<double> mask;
        auto m_var = tensor_tree::get_var(var_tree->children[0]);
        autodiff::computation_graph& graph = *m_var->graph;
        la::tensor<double>& m = autodiff::get_output<la::tensor<double>>(m_var);
        mask.resize({m.size(1)});

        std::bernoulli_distribution dist { 1.0 - prob };

        for (int i = 0; i < m.size(1); ++i) {
            mask({i}) = dist(gen) / (1.0 - prob);
        }

        std::shared_ptr<autodiff::op_t> output = (*base)(var_tree, input);

        auto mask_var = graph.var(std::move(mask));

        return autodiff::emul(mask_var, output);
    }

    dyer_lstm_step_transcriber::dyer_lstm_step_transcriber()
        : cell(nullptr), output(nullptr)
    {}

    std::shared_ptr<autodiff::op_t> dyer_lstm_step_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        lstm_step_nn_t nn = make_dyer_lstm_step_nn(var_tree, cell, output, input);

        cell = nn.cell;
        output = nn.output;

        return output;
    }

    std::shared_ptr<autodiff::op_t> lstm_multistep_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        for (int i = 0; i < steps.size(); ++i) {
            input = (*steps.at(i))(var_tree->children[i], input);
        }

        return input;
    }

    transcriber::~transcriber()
    {}

    lstm_transcriber::lstm_transcriber(
        std::shared_ptr<step_transcriber> step)
        : step(step)
    {}

    std::vector<std::shared_ptr<autodiff::op_t>>
    lstm_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        for (int i = 0; i < feat.size(); ++i) {
            std::shared_ptr<autodiff::op_t> output = (*step)(var_tree, feat[i]);
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
                autodiff::mul(forward[i], get_var(var_tree->children[2])),
                autodiff::mul(backward[i], get_var(var_tree->children[3])),
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

    logsoftmax_transcriber::logsoftmax_transcriber(
        std::shared_ptr<transcriber> base)
        : base(base)
    {}

    std::vector<std::shared_ptr<autodiff::op_t>> logsoftmax_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> output;

        if (base == nullptr) {
            output = feat;
        } else {
            output = (*base)(var_tree->children[0], feat);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> result;

        for (int i = 0; i < output.size(); ++i) {
            result.push_back(autodiff::logsoftmax(
                autodiff::add(
                    autodiff::mul(output[i], tensor_tree::get_var(var_tree->children[1])),
                    tensor_tree::get_var(var_tree->children[2])
                )
            ));
        }

        return result;
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

    std::vector<std::shared_ptr<autodiff::op_t>> hypercolumn_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> const *input = &feat;
        std::vector<std::shared_ptr<autodiff::op_t>> output;

        std::vector<std::shared_ptr<autodiff::op_t>> result;

        for (int j = 0; j < feat.size(); ++j) {
            result.push_back(autodiff::add(
                autodiff::mul(feat.at(j),
                    tensor_tree::get_var(var_tree->children[1])),
                tensor_tree::get_var(var_tree->children.back())
            ));
        }

        result.resize(feat.size());

        for (int i = 0; i < layer.size(); ++i) {
            output = (*layer[i])(var_tree->children[0]->children[i], *input);

            int freq = std::round(double(feat.size()) / output.size());

            for (int j = 0; j < feat.size(); ++j) {
                result.at(j) = autodiff::add(
                    autodiff::mul(output.at(j / freq),
                        tensor_tree::get_var(var_tree->children[2 + i])),
                    result.at(j)
                );
            }

            input = &output;
        }

        return result;
    }

}
