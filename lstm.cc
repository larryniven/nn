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

        autodiff::computation_graph& graph = *tensor_tree::get_var(var_tree->children[0])->graph;
        la::tensor<double>& h_vec = autodiff::get_output<la::tensor<double>>(
            tensor_tree::get_var(var_tree->children[2]));
        unsigned int h_dim = h_vec.vec_size();

        result.input = nullptr;

        std::vector<std::shared_ptr<autodiff::op_t>> h_comp { get_var(var_tree->children[2]) };
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp { get_var(var_tree->children[6]) };

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

        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));
        result.input_gate = autodiff::logistic(autodiff::add(input_gate_comp));

        if (cell != nullptr) {
            la::tensor<double> one_vec;
            one_vec.resize({ h_dim }, 1);
            auto one = graph.var(std::move(one_vec));

            result.forget_gate = autodiff::sub(one, result.input_gate);

            result.cell = autodiff::add(
                autodiff::emul(result.forget_gate, cell),
                autodiff::emul(result.input_gate, h));
        } else {
            result.cell = autodiff::emul(result.input_gate, h);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_comp {
            get_var(var_tree->children[10]), autodiff::mul(result.cell, get_var(var_tree->children[9])) };

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
                autodiff::mul(result.forward_nn.output[i], get_var(var_tree->children[2])),
                autodiff::mul(result.backward_nn.output[i], get_var(var_tree->children[3])),
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
        unsigned int dim,
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
            la::tensor<double> v;
            v.resize({dim});
            for (int d = 0; d < v.vec_size(); ++d) {
                v({d}) = bernoulli(gen) / (1.0 - prob);
            }
            std::shared_ptr<autodiff::op_t> input_mask = comp_graph.var(std::move(v));
            masked_input.push_back(autodiff::emul(feat[j], input_mask));
        }

        return (*builder)(var_tree, masked_input);
    }

    bi_lstm_input_scaling::bi_lstm_input_scaling(
        autodiff::computation_graph& comp_graph, unsigned int dim, double scale,
        std::shared_ptr<bi_lstm_builder> builder)
        : comp_graph(comp_graph), dim(dim), scale(scale), builder(builder)
    {}

    bi_lstm_nn_t bi_lstm_input_scaling::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::shared_ptr<autodiff::op_t>> const& feat) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> masked_input;

        for (int j = 0; j < feat.size(); ++j) {
            la::tensor<double> v;
            v.resize({dim}, scale);
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
