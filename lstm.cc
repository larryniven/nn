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
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> cell_mask)
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

        if (cell_mask != nullptr) {
            result.cell = autodiff::emul(result.cell, autodiff::rep_col_to(cell_mask, result.cell));
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
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> cell_mask)
    {
        // TODO: minibatching does not work here
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
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> cell_mask)
    {
        auto mask = autodiff::dropout_mask(input, prob, gen);

        return (*base)(var_tree, autodiff::emul(mask, input), cell_mask);
    }

    output_dropout_transcriber::output_dropout_transcriber(
        std::default_random_engine& gen, double prob,
        std::shared_ptr<step_transcriber> base)
        : gen(gen), prob(prob), base(base)
    {}

    std::shared_ptr<autodiff::op_t> output_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> cell_mask)
    {
        std::shared_ptr<autodiff::op_t> output = (*base)(var_tree, input, cell_mask);

        auto mask = autodiff::dropout_mask(output, prob, gen);

        return autodiff::emul(mask, output);
    }

    dyer_lstm_step_transcriber::dyer_lstm_step_transcriber()
        : cell(nullptr), output(nullptr)
    {}

    std::shared_ptr<autodiff::op_t> dyer_lstm_step_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> cell_mask)
    {
        lstm_step_nn_t nn = make_dyer_lstm_step_nn(var_tree, cell, output, input, cell_mask);

        cell = nn.cell;
        output = nn.output;

        return output;
    }

    std::shared_ptr<autodiff::op_t> lstm_multistep_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> cell_mask)
    {
        for (int i = 0; i < steps.size(); ++i) {
            input = (*steps.at(i))(var_tree->children[i], input, cell_mask);
        }

        return input;
    }

    transcriber::~transcriber()
    {}

    std::vector<std::shared_ptr<autodiff::op_t>>
    transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> mask;
        mask.resize(feat.size(), nullptr);

        std::vector<std::shared_ptr<autodiff::op_t>> result;
        std::vector<std::shared_ptr<autodiff::op_t>> o_mask;
        std::tie(result, o_mask) = (*this)(var_tree, feat, mask);

        return result;
    }

    lstm_transcriber::lstm_transcriber(
        std::shared_ptr<step_transcriber> step)
        : step(step)
    {}

    std::pair<std::vector<std::shared_ptr<autodiff::op_t>>,
        std::vector<std::shared_ptr<autodiff::op_t>>>
    lstm_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        for (int i = 0; i < feat.size(); ++i) {
            std::shared_ptr<autodiff::op_t> output = (*step)(var_tree, feat[i], mask[i]);
            result.push_back(output);
        }

        return std::make_pair(result, mask);
    }

    bi_transcriber::bi_transcriber(std::shared_ptr<transcriber> base)
        : base(base)
    {}

    std::pair<std::vector<std::shared_ptr<autodiff::op_t>>,
        std::vector<std::shared_ptr<autodiff::op_t>>>
    bi_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        std::vector<std::shared_ptr<autodiff::op_t>> f = feat;
        std::vector<std::shared_ptr<autodiff::op_t>> m = mask;

        std::vector<std::shared_ptr<autodiff::op_t>> f_output_mask;
        std::vector<std::shared_ptr<autodiff::op_t>> b_output_mask;

        std::vector<std::shared_ptr<autodiff::op_t>> forward;
        std::vector<std::shared_ptr<autodiff::op_t>> backward;

        std::tie(forward, f_output_mask) = (*base)(var_tree->children[0], f, m);

        std::reverse(f.begin(), f.end());
        std::reverse(m.begin(), m.end());

        std::tie(backward, b_output_mask) = (*base)(var_tree->children[1], f, m);

        assert(f_output_mask.size() == b_output_mask.size());

        std::reverse(backward.begin(), backward.end());

        std::vector<std::vector<std::shared_ptr<autodiff::op_t>>> comps;

        for (int i = 0; i < forward.size(); ++i) {
            std::vector<std::shared_ptr<autodiff::op_t>> comp {
                autodiff::mul(forward[i], get_var(var_tree->children[2])),
                autodiff::mul(backward[i], get_var(var_tree->children[3])),
            };
            comps.push_back(comp);
        }

        auto output_bias = autodiff::rep_row_to(get_var(var_tree->children[4]), comps.back().back());

        for (int i = 0; i < forward.size(); ++i) {
            comps[i].push_back(output_bias);
            result.push_back(autodiff::add(comps[i]));
        }

        return std::make_pair(result, f_output_mask);
    }

    std::pair<std::vector<std::shared_ptr<autodiff::op_t>>,
        std::vector<std::shared_ptr<autodiff::op_t>>>
    layered_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> output = feat;
        std::vector<std::shared_ptr<autodiff::op_t>> o_mask = mask;

        for (int i = 0; i < layer.size(); ++i) {
            std::tie(output, o_mask) = (*layer[i])(var_tree->children[i], output, o_mask);
        }

        return std::make_pair(output, o_mask);
    }

    logsoftmax_transcriber::logsoftmax_transcriber(
        std::shared_ptr<transcriber> base)
        : base(base)
    {}

    std::pair<std::vector<std::shared_ptr<autodiff::op_t>>,
        std::vector<std::shared_ptr<autodiff::op_t>>>
    logsoftmax_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> output;
        std::vector<std::shared_ptr<autodiff::op_t>> o_mask;

        if (base == nullptr) {
            output = feat;
            o_mask = mask;
        } else {
            std::tie(output, o_mask) = (*base)(var_tree->children[0], feat, mask);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> result;

        for (int i = 0; i < output.size(); ++i) {
            auto h = autodiff::mul(output[i], tensor_tree::get_var(var_tree->children[1]));
            auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[2]), h);

            result.push_back(autodiff::logsoftmax(
                autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>>{h, b})
            ));
        }

        return std::make_pair(result, o_mask);
    }

    subsampled_transcriber::subsampled_transcriber(
        int freq, int shift, std::shared_ptr<transcriber> base)
        : freq(freq), shift(shift), base(base)
    {}

    std::pair<std::vector<std::shared_ptr<autodiff::op_t>>,
        std::vector<std::shared_ptr<autodiff::op_t>>>
    subsampled_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat,
        std::vector<std::shared_ptr<autodiff::op_t>> const& mask) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;
        std::vector<std::shared_ptr<autodiff::op_t>> output;
        std::vector<std::shared_ptr<autodiff::op_t>> o_mask;
        std::tie(output, o_mask) = (*base)(var_tree, feat, mask);

        std::vector<std::shared_ptr<autodiff::op_t>> subsampled_mask;

        for (int i = 0; i < output.size(); ++i) {
            if ((i - shift) % freq == 0) {
                result.push_back(output[i]);
                subsampled_mask.push_back(o_mask[i]);
            }
        }

        return std::make_pair(result, subsampled_mask);
    }

}
