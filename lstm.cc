#include "nn/lstm.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>

namespace lstm {

    lstm_step_nn_t make_lstm_step_nn(
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> prev_output,
        std::shared_ptr<autodiff::op_t> prev_cell,
        std::shared_ptr<autodiff::op_t> output_weight,
        std::shared_ptr<autodiff::op_t> cell2i,
        std::shared_ptr<autodiff::op_t> cell2f,
        std::shared_ptr<autodiff::op_t> cell2o,
        std::shared_ptr<autodiff::op_t> cell_mask,
        std::shared_ptr<autodiff::op_t> output_storage,
        int batch_size,
        int cell_dim)
    {
        lstm_step_nn_t result;

        std::shared_ptr<autodiff::op_t> pre_gates;

        if (prev_output == nullptr) {
            pre_gates = input;
        } else {
            pre_gates = autodiff::add(
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    input,
                    autodiff::mul(prev_output, output_weight)
                });
        }

        unsigned int ubatch_size = batch_size;
        unsigned int ucell_dim = cell_dim;

        auto pre_g = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, 0},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_i = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, ucell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_f = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, 2 * ucell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_o = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, 3 * ucell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto g = autodiff::tanh(pre_g);
        if (prev_cell == nullptr) {
            result.input_gate = autodiff::logistic(pre_i);
            result.forget_gate = autodiff::logistic(pre_f);
            result.cell = autodiff::emul(result.input_gate, g);
        } else {
            result.input_gate = autodiff::logistic(autodiff::add(pre_i,
                autodiff::emul(prev_cell, cell2i)));
            result.forget_gate = autodiff::logistic(autodiff::add(pre_f,
                autodiff::emul(prev_cell, cell2f)));
            result.cell = autodiff::add(autodiff::emul(result.input_gate, g),
                autodiff::emul(result.forget_gate, prev_cell));
        }

        if (cell_mask != nullptr) {
            result.cell = autodiff::emul(result.cell,
                autodiff::rep_col_to(cell_mask, result.cell));
        }

        result.output_gate = autodiff::logistic(
            autodiff::add(pre_o, autodiff::emul(result.cell, cell2o)));

        result.output = autodiff::emul_to(output_storage,
            result.output_gate, autodiff::tanh(result.cell));

        return result;
    }

    lstm_step_nn_t make_dyer_lstm_step_nn(
        std::shared_ptr<autodiff::op_t> input,
        std::shared_ptr<autodiff::op_t> prev_output,
        std::shared_ptr<autodiff::op_t> prev_cell,
        std::shared_ptr<autodiff::op_t> output_weight,
        std::shared_ptr<autodiff::op_t> cell2i,
        std::shared_ptr<autodiff::op_t> cell2o,
        std::shared_ptr<autodiff::op_t> cell_mask,
        std::shared_ptr<autodiff::op_t> output_storage,
        int batch_size,
        int cell_dim)
    {
        lstm_step_nn_t result;

        auto pre_gates = autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>> {
                input,
                autodiff::mul(prev_output, output_weight)
            });

        unsigned int ubatch_size = batch_size;
        unsigned int ucell_dim = cell_dim;

        auto pre_g = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, 0},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_i = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, ucell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_o = autodiff::subtensor(pre_gates, std::vector<unsigned int> {0, 2 * ucell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto g = autodiff::tanh(pre_g);
        result.input_gate = autodiff::logistic(autodiff::add(pre_i, autodiff::mul(prev_cell, cell2i)));

        auto one = autodiff::resize_as(result.input_gate, 1);
        one->grad_needed = false;

        result.forget_gate = autodiff::sub(one, result.input_gate);

        result.cell = autodiff::add(autodiff::emul(result.input_gate, g),
            autodiff::emul(result.forget_gate, prev_cell));

        if (cell_mask != nullptr) {
            result.cell = autodiff::emul(result.cell,
                autodiff::rep_col_to(cell_mask, result.cell));
        }

        result.output_gate = autodiff::logistic(
            autodiff::add(pre_o, autodiff::mul(result.cell, cell2o)));

        result.output = autodiff::emul_to(output_storage,
            result.output_gate, autodiff::tanh(result.cell));

        return result;
    }

    // trans_seq_t

    trans_seq_t make_trans_seq(std::shared_ptr<autodiff::op_t> var)
    {
        trans_seq_t result;

        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(var);

        assert(t.dim() == 3);

        result.nframes = t.size(0);
        result.batch_size = t.size(1);
        result.dim = t.size(2);
        result.feat = var;
        result.mask = nullptr;

        return result;
    }

    // transcriber

    transcriber::~transcriber()
    {}

    bool transcriber::require_param() const
    {
        return true;
    }

    lstm_transcriber::lstm_transcriber(int cell_dim, bool reverse)
        : cell_dim(cell_dim), reverse(reverse)
    {}

    trans_seq_t lstm_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result;

        result.nframes = seq.nframes;
        result.batch_size = seq.batch_size;
        result.mask = seq.mask;
        result.dim = cell_dim;

        auto pre_input = autodiff::mul(seq.feat, tensor_tree::get_var(var_tree->children[0]));
        auto bias = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), pre_input);
        auto input = autodiff::add(pre_input, bias);

        unsigned int unframes = seq.nframes;
        unsigned int ubatch_size = seq.batch_size;
        unsigned int ucell_dim = cell_dim;

        auto& comp_graph = *seq.feat->graph;

        la::cpu::tensor<double> zeros;
        zeros.resize(std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto cell = comp_graph.var(zeros);
        auto output = comp_graph.var(zeros);

        zeros.resize(std::vector<unsigned int> { unframes, ubatch_size, ucell_dim });

        auto output_storage = comp_graph.var(zeros);

        std::vector<std::shared_ptr<autodiff::op_t>> outputs;

        if (reverse) {
            for (int t = seq.nframes - 1; t >= 0; --t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (seq.mask != nullptr) {
                    mask_t = autodiff::weak_var(seq.mask, t * seq.batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * seq.batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * seq.batch_size * 4 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 4 * ucell_dim});

                lstm_step_nn_t result = make_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    tensor_tree::get_var(var_tree->children[5]),
                    mask_t, output_t_storage,
                    seq.batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }

            std::reverse(outputs.begin(), outputs.end());
        } else {
            for (int t = 0; t < seq.nframes; ++t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (seq.mask != nullptr) {
                    mask_t = autodiff::weak_var(seq.mask, t * seq.batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * seq.batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * seq.batch_size * 4 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 4 * ucell_dim});

                lstm_step_nn_t result = make_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    tensor_tree::get_var(var_tree->children[5]),
                    mask_t, output_t_storage,
                    seq.batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }
        }

        result.feat = autodiff::weak_cat(outputs, output_storage);

        return result;
    }

    dyer_lstm_transcriber::dyer_lstm_transcriber(
        int cell_dim, bool reverse)
        : cell_dim(cell_dim), reverse(reverse)
    {}

    trans_seq_t dyer_lstm_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result;
        result.nframes = seq.nframes;
        result.batch_size = seq.batch_size;
        result.mask = seq.mask;
        result.dim = cell_dim;

        auto pre_input = autodiff::mul(seq.feat, tensor_tree::get_var(var_tree->children[0]));
        auto bias = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), pre_input);
        auto input = autodiff::add(pre_input, bias);

        unsigned int unframes = seq.nframes;
        unsigned int ubatch_size = seq.batch_size;
        unsigned int ucell_dim = cell_dim;

        auto& comp_graph = *seq.feat->graph;

        la::cpu::tensor<double> zeros;
        zeros.resize(std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto cell = comp_graph.var(zeros);
        auto output = comp_graph.var(zeros);

        zeros.resize(std::vector<unsigned int> { unframes, ubatch_size, ucell_dim });

        auto output_storage = comp_graph.var(zeros);

        std::vector<std::shared_ptr<autodiff::op_t>> outputs;

        if (reverse) {
            for (int t = seq.nframes - 1; t >= 0; --t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (seq.mask != nullptr) {
                    mask_t = autodiff::weak_var(seq.mask, t * seq.batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * seq.batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * seq.batch_size * 3 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 3 * ucell_dim});

                lstm_step_nn_t result = make_dyer_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    mask_t, output_t_storage,
                    seq.batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }

            std::reverse(outputs.begin(), outputs.end());
        } else {
            for (int t = 0; t < seq.nframes; ++t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (seq.mask != nullptr) {
                    mask_t = autodiff::weak_var(seq.mask, t * seq.batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * seq.batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * seq.batch_size * 3 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 3 * ucell_dim});

                lstm_step_nn_t result = make_dyer_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    mask_t, output_t_storage,
                    seq.batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }
        }

        result.feat = autodiff::weak_cat(outputs, output_storage);

        return result;
    }

    input_dropout_transcriber::input_dropout_transcriber(
        std::shared_ptr<transcriber> base,
        double prob,
        std::default_random_engine& gen)
        : base(base), prob(prob), gen(gen)
    {}

    trans_seq_t input_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t input_seq = seq;

        auto d_mask = autodiff::dropout_mask(seq.feat, prob, gen);
        input_seq.feat = autodiff::emul(seq.feat, d_mask);

        return (*base)(var_tree, input_seq);
    }

    output_dropout_transcriber::output_dropout_transcriber(
        std::shared_ptr<transcriber> base,
        double prob,
        std::default_random_engine& gen)
        : base(base), prob(prob), gen(gen)
    {}

    trans_seq_t output_dropout_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result = (*base)(var_tree, seq);

        auto d_mask = autodiff::dropout_mask(result.feat, prob, gen);
        result.feat = autodiff::emul(d_mask, result.feat);

        return result;
    }

    bi_transcriber::bi_transcriber(int output_dim,
        std::shared_ptr<transcriber> f_base,
        std::shared_ptr<transcriber> b_base)
        : output_dim(output_dim), f_base(f_base), b_base(b_base)
    {}

    trans_seq_t bi_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t seq1 = (*f_base)(var_tree->children[0], seq);
        trans_seq_t seq2 = (*b_base)(var_tree->children[1], seq);

        assert(seq1.nframes == seq2.nframes);
        assert(seq1.batch_size == seq2.batch_size);

        auto h = autodiff::mul(seq1.feat, tensor_tree::get_var(var_tree->children[2]));

        trans_seq_t result = seq1;
        result.dim = output_dim;

        result.feat = autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>> {
                h,
                autodiff::mul(seq2.feat, tensor_tree::get_var(var_tree->children[3])),
                autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[4]), h)
            }
        );

        return result;
    }

    trans_seq_t layered_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result = seq;

        int j = 0;
        for (int i = 0; i < layer.size(); ++i) {
            if (layer[i]->require_param()) {
                result = (*layer[i])(var_tree->children[j], result);
                ++j;
            } else {
                result = (*layer[i])(nullptr, result);
            }
        }

        return result;
    }

    fc_transcriber::fc_transcriber(
        int output_dim)
        : output_dim(output_dim)
    {}

    trans_seq_t fc_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result = seq;
        result.dim = output_dim;

        auto h = autodiff::mul(seq.feat, tensor_tree::get_var(var_tree->children[0]));

        result.feat = autodiff::add(
            h,
            autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), h)
        );

        return result;
    }

    bool logsoftmax_transcriber::require_param() const
    {
        return false;
    }

    trans_seq_t logsoftmax_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result = seq;
        result.feat = autodiff::logsoftmax(seq.feat);

        return result;
    }

    res_transcriber::res_transcriber(std::shared_ptr<transcriber> base)
        : base(base)
    {}

    trans_seq_t res_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        trans_seq_t result = (*base)(var_tree, seq);

        result.feat = autodiff::add(result.feat, seq.feat);

        return result;
    }

    subsampled_transcriber::subsampled_transcriber(
        int freq, int shift)
        : freq(freq), shift(shift)
    {}

    bool subsampled_transcriber::require_param() const
    {
        return false;
    }

    trans_seq_t subsampled_transcriber::operator()(
        std::shared_ptr<tensor_tree::vertex> var_tree,
        trans_seq_t const& seq) const
    {
        std::vector<std::shared_ptr<autodiff::op_t>> subsamp_feat;
        std::vector<std::shared_ptr<autodiff::op_t>> subsamp_mask;

        unsigned int ubatch_size = seq.batch_size;
        unsigned int udim = seq.dim;

        for (int i = 0; i < seq.nframes; ++i) {
            if ((i - shift) % freq == 0) {
                subsamp_feat.push_back(autodiff::weak_var(seq.feat,
                    i * seq.batch_size * seq.dim,
                    std::vector<unsigned int> { ubatch_size, udim }));

                if (seq.mask != nullptr) {
                    subsamp_mask.push_back(autodiff::weak_var(seq.mask,
                        i * seq.batch_size,
                        std::vector<unsigned int> { ubatch_size }));
                }
            }
        }

        trans_seq_t result;

        result.batch_size = seq.batch_size;
        result.dim = seq.dim;
        result.feat = autodiff::row_cat(subsamp_feat);
        result.nframes = subsamp_feat.size();

        if (seq.mask != nullptr) {
            result.mask = autodiff::row_cat(subsamp_mask);
        }

        return result;
    }

}
