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

        auto pre_gates = autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>> {
                input,
                autodiff::mul(prev_output, output_weight)
            });

        unsigned int ubatch_size = batch_size;
        unsigned int ucell_dim = cell_dim;

        auto pre_g = autodiff::subtensor(pre_gates, std::vector<int> {0, 0},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_i = autodiff::subtensor(pre_gates, std::vector<int> {0, cell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_f = autodiff::subtensor(pre_gates, std::vector<int> {0, 2 * cell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_o = autodiff::subtensor(pre_gates, std::vector<int> {0, 3 * cell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto g = autodiff::tanh(pre_g);
        result.input_gate = autodiff::logistic(autodiff::add(pre_i, autodiff::emul(prev_cell, cell2i)));
        result.forget_gate = autodiff::logistic(autodiff::add(pre_f, autodiff::emul(prev_cell, cell2f)));

        result.cell = autodiff::add(autodiff::emul(result.input_gate, g),
            autodiff::emul(result.forget_gate, prev_cell));

        if (cell_mask != nullptr) {
            result.cell = autodiff::emul(result.cell,
                autodiff::rep_col_to(cell_mask, result.cell));
        }

        result.output_gate = autodiff::logistic(
            autodiff::add(pre_o, autodiff::emul(result.cell, cell2o)));

        result.output = autodiff::emul(output_storage,
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

        auto pre_g = autodiff::subtensor(pre_gates, std::vector<int> {0, 0},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_i = autodiff::subtensor(pre_gates, std::vector<int> {0, cell_dim},
            std::vector<unsigned int> { ubatch_size, ucell_dim });
        auto pre_o = autodiff::subtensor(pre_gates, std::vector<int> {0, 2 * cell_dim},
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

        result.output = autodiff::emul(output_storage,
            result.output_gate, autodiff::tanh(result.cell));

        return result;
    }

#if 0
    lstm_step_nn_t make_dyer_lstm_step_nn(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> prev_cell,
        std::shared_ptr<autodiff::op_t> prev_output,
        std::shared_ptr<autodiff::op_t> input_h,
        std::shared_ptr<autodiff::op_t> input_i,
        std::shared_ptr<autodiff::op_t> input_o,
        std::shared_ptr<autodiff::op_t> cell,
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> cell_mask)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> h_comp;
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_comp;

        if (input_h != nullptr) {
            h_comp.push_back(input_h);
        }

        if (input_i != nullptr) {
            input_gate_comp.push_back(input_i);
        }

        if (prev_output != nullptr) {
            h_comp.push_back(autodiff::mul(prev_output, get_var(var_tree->children[1])));
            input_gate_comp.push_back(autodiff::mul(prev_output, get_var(var_tree->children[4])));
        }

        if (prev_cell != nullptr) {
            input_gate_comp.push_back(autodiff::mul(prev_cell, get_var(var_tree->children[5])));
        }

        assert(h_comp.size() > 0);
        h_comp.push_back(autodiff::rep_row_to(get_var(var_tree->children[2]), h_comp.back()));
        std::shared_ptr<autodiff::op_t> h = autodiff::tanh(autodiff::add(h_comp));

        assert(input_gate_comp.size() > 0);
        input_gate_comp.push_back(autodiff::rep_row_to(get_var(var_tree->children[6]),
            input_gate_comp.back()));
        auto input_gate = autodiff::logistic(autodiff::add(input_gate_comp));

        std::shared_ptr<autodiff::op_t> forget_gate = nullptr;
        std::shared_ptr<autodiff::op_t> cell_tmp = nullptr;

        if (prev_cell != nullptr) {
            auto one = autodiff::resize_as(input_gate, 1);
            one->grad_needed = false;

            forget_gate = autodiff::sub(one, input_gate);

            cell_tmp = autodiff::add(
                autodiff::emul(forget_gate, prev_cell),
                autodiff::emul(input_gate, h));
        } else {
            cell_tmp = autodiff::emul(input_gate, h);
        }

        if (cell_mask != nullptr) {
            cell_tmp = autodiff::emul(cell_tmp, autodiff::rep_col_to(cell_mask, cell_tmp));
        }

        autodiff::add(cell, std::vector<std::shared_ptr<autodiff::op_t>>{cell_tmp});

        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_comp {
            autodiff::mul(cell, get_var(var_tree->children[9])) };

        output_gate_comp.push_back(autodiff::rep_row_to(get_var(var_tree->children[10]),
            output_gate_comp.back()));

        if (input_o != nullptr) {
            output_gate_comp.push_back(input_o);
        }

        if (prev_output != nullptr) {
            output_gate_comp.push_back(autodiff::mul(prev_output, get_var(var_tree->children[8])));
        }

        auto output_gate = autodiff::logistic(autodiff::add(output_gate_comp));

        autodiff::emul(output, output_gate, autodiff::tanh(cell));

        lstm_step_nn_t result;

        result.input_gate = input_gate;
        result.output_gate = output_gate;
        result.forget_gate = forget_gate;
        result.cell = cell;
        result.output = output;

        return result;
    }
#endif

    // transcriber

    transcriber::~transcriber()
    {}

    lstm_transcriber::lstm_transcriber(bool reverse)
        : reverse(reverse)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    lstm_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        auto pre_input = autodiff::mul(feat, tensor_tree::get_var(var_tree->children[0]));
        auto bias = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), pre_input);
        auto input = autodiff::add(pre_input, bias);

        unsigned int unframes = nframes;
        unsigned int ubatch_size = batch_size;
        unsigned int ucell_dim = cell_dim;

        auto& comp_graph = *feat->graph;

        la::cpu::tensor<double> zeros;
        zeros.resize(std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto cell = comp_graph.var(zeros);
        auto output = comp_graph.var(zeros);

        zeros.resize(std::vector<unsigned int> { unframes, ubatch_size, ucell_dim });

        auto output_storage = comp_graph.var(zeros);

        std::vector<std::shared_ptr<autodiff::op_t>> outputs;

        if (reverse) {
            for (int t = nframes - 1; t >= 0; --t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (mask != nullptr) {
                    mask_t = autodiff::weak_var(mask, t * batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * batch_size * 4 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 4 * ucell_dim});

                lstm_step_nn_t result = make_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    tensor_tree::get_var(var_tree->children[5]),
                    mask_t, output_t_storage,
                    batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }

            std::reverse(outputs.begin(), outputs.end());
        } else {
            for (int t = 0; t < nframes; ++t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (mask != nullptr) {
                    mask_t = autodiff::weak_var(mask, t * batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * batch_size * 4 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 4 * ucell_dim});

                lstm_step_nn_t result = make_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    tensor_tree::get_var(var_tree->children[5]),
                    mask_t, output_t_storage,
                    batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }
        }

        return std::make_pair(autodiff::weak_cat(outputs, output_storage), mask);
    }

    // eager only
    std::vector<std::shared_ptr<autodiff::op_t>> split_rows(
        std::shared_ptr<autodiff::op_t> t)
    {
        auto& v = autodiff::get_output<la::tensor_like<double>>(t);
        std::vector<std::shared_ptr<autodiff::op_t>> result;

        std::vector<unsigned int> sizes = v.sizes();
        std::vector<unsigned int> new_sizes { sizes.begin() + 1, sizes.end() };
        int new_vec_size = v.vec_size() / v.size(0);

        for (int i = 0; i < v.size(0); ++i) {
            result.push_back(autodiff::weak_var(t, new_vec_size * i, new_sizes));
        }

        return result;
    }

    dyer_lstm_transcriber::dyer_lstm_transcriber(
        bool reverse)
        : reverse(reverse)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    dyer_lstm_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        auto pre_input = autodiff::mul(feat, tensor_tree::get_var(var_tree->children[0]));
        auto bias = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), pre_input);
        auto input = autodiff::add(pre_input, bias);

        unsigned int unframes = nframes;
        unsigned int ubatch_size = batch_size;
        unsigned int ucell_dim = cell_dim;

        auto& comp_graph = *feat->graph;

        la::cpu::tensor<double> zeros;
        zeros.resize(std::vector<unsigned int> { ubatch_size, ucell_dim });

        auto cell = comp_graph.var(zeros);
        auto output = comp_graph.var(zeros);

        zeros.resize(std::vector<unsigned int> { unframes, ubatch_size, ucell_dim });

        auto output_storage = comp_graph.var(zeros);

        std::vector<std::shared_ptr<autodiff::op_t>> outputs;

        if (reverse) {
            for (int t = nframes - 1; t >= 0; --t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (mask != nullptr) {
                    mask_t = autodiff::weak_var(mask, t * batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * batch_size * 3 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 3 * ucell_dim});

                lstm_step_nn_t result = make_dyer_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    mask_t, output_t_storage,
                    batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }

            std::reverse(outputs.begin(), outputs.end());
        } else {
            for (int t = 0; t < nframes; ++t) {
                std::shared_ptr<autodiff::op_t> mask_t = nullptr;

                if (mask != nullptr) {
                    mask_t = autodiff::weak_var(mask, t * batch_size,
                        std::vector<unsigned int> { ubatch_size });
                }

                auto output_t_storage = autodiff::weak_var(output_storage, t * batch_size * cell_dim,
                    std::vector<unsigned int> { ubatch_size, ucell_dim });

                auto input_t = autodiff::weak_var(input, t * batch_size * 3 * cell_dim,
                    std::vector<unsigned int> { ubatch_size, 3 * ucell_dim});

                lstm_step_nn_t result = make_dyer_lstm_step_nn(input_t, output, cell,
                    tensor_tree::get_var(var_tree->children[2]),
                    tensor_tree::get_var(var_tree->children[3]),
                    tensor_tree::get_var(var_tree->children[4]),
                    mask_t, output_t_storage,
                    batch_size, cell_dim);

                cell = result.cell;
                output = result.output;

                outputs.push_back(output);
            }
        }

        return std::make_pair(autodiff::weak_cat(outputs, output_storage), mask);
    }

#if 0
    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    dyer_lstm_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        auto h = autodiff::mul(feat, tensor_tree::get_var(var_tree->children[0]));

        auto input_h = autodiff::add(h,
            autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[2]), h));

        auto input_h_vars = split_rows(input_h);

        auto input_i = autodiff::add(autodiff::mul(feat, tensor_tree::get_var(var_tree->children[3])),
            autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[6]), h));

        auto input_i_vars = split_rows(input_i);

        auto input_o = autodiff::add(autodiff::mul(feat, tensor_tree::get_var(var_tree->children[7])),
            autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[10]), h));

        auto input_o_vars = split_rows(input_o);

        std::shared_ptr<autodiff::op_t> cell = autodiff::resize_as(input_h);
        std::vector<std::shared_ptr<autodiff::op_t>> cell_vars = split_rows(cell);

        auto output = autodiff::resize_as(input_h);
        std::vector<std::shared_ptr<autodiff::op_t>> output_vars = split_rows(output);

        std::vector<std::shared_ptr<autodiff::op_t>> mask_vars;
        if (mask == nullptr) {
            mask_vars.resize(output_vars.size(), nullptr);
        } else {
            mask_vars = split_rows(mask);
        }

        if (reverse) {
            for (int i = nframes - 1; i >= 0; --i) {
                if (i == nframes - 1) {
                    make_dyer_lstm_step_nn(var_tree, nullptr, nullptr,
                       input_h_vars[i], input_i_vars[i], input_o_vars[i],
                       cell_vars[i], output_vars[i], mask_vars[i]);
                } else {
                    make_dyer_lstm_step_nn(var_tree, cell_vars[i+1], output_vars[i+1],
                       input_h_vars[i], input_i_vars[i], input_o_vars[i],
                       cell_vars[i], output_vars[i], mask_vars[i]);
                }
            }
        } else {
            for (int i = 0; i < nframes; ++i) {
                if (i == 0) {
                    make_dyer_lstm_step_nn(var_tree, nullptr, nullptr,
                       input_h_vars[i], input_i_vars[i], input_o_vars[i],
                       cell_vars[i], output_vars[i], mask_vars[i]);
                } else {
                    make_dyer_lstm_step_nn(var_tree, cell_vars[i-1], output_vars[i-1],
                       input_h_vars[i], input_i_vars[i], input_o_vars[i],
                       cell_vars[i], output_vars[i], mask_vars[i]);
                }
            }
        }

        return std::make_pair(output, mask);
    }
#endif

    input_dropout_transcriber::input_dropout_transcriber(
        std::shared_ptr<transcriber> base,
        double prob,
        std::default_random_engine& gen)
        : base(base), prob(prob), gen(gen)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    input_dropout_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        auto d_mask = autodiff::dropout_mask(feat, prob, gen);

        return (*base)(nframes, batch_size, cell_dim, var_tree, autodiff::emul(feat, d_mask), mask);
    }

    output_dropout_transcriber::output_dropout_transcriber(
        std::shared_ptr<transcriber> base,
        double prob,
        std::default_random_engine& gen)
        : base(base), prob(prob), gen(gen)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    output_dropout_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> o_mask;

        std::tie(output, o_mask) = (*base)(nframes, batch_size, cell_dim, var_tree, feat, mask);

        auto d_mask = autodiff::dropout_mask(output, prob, gen);

        return std::make_pair(autodiff::emul(d_mask, output), o_mask);
    }

    bi_transcriber::bi_transcriber(std::shared_ptr<transcriber> f_base,
        std::shared_ptr<transcriber> b_base)
        : f_base(f_base), b_base(b_base)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    bi_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        std::shared_ptr<autodiff::op_t> f_mask;
        std::shared_ptr<autodiff::op_t> f_output;
        std::shared_ptr<autodiff::op_t> b_mask;
        std::shared_ptr<autodiff::op_t> b_output;

        std::tie(f_output, f_mask) = (*f_base)(nframes, batch_size, cell_dim, var_tree->children[0], feat, mask);
        std::tie(b_output, b_mask) = (*b_base)(nframes, batch_size, cell_dim, var_tree->children[1], feat, mask);

        auto h = autodiff::mul(f_output, tensor_tree::get_var(var_tree->children[2]));

        auto result = autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>> {
                h,
                autodiff::mul(b_output, tensor_tree::get_var(var_tree->children[3])),
                autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[4]), h)
            }
        );

        return std::make_pair(result, f_mask);
    }

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    layered_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        std::shared_ptr<autodiff::op_t> output = feat;
        std::shared_ptr<autodiff::op_t> o_mask = mask;

        for (int i = 0; i < layer.size(); ++i) {
            std::tie(output, o_mask) = (*layer[i])(nframes, batch_size, cell_dim, var_tree->children[i], output, o_mask);
        }

        return std::make_pair(output, o_mask);
    }

    logsoftmax_transcriber::logsoftmax_transcriber(
        std::shared_ptr<transcriber> base)
        : base(base)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    logsoftmax_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> o_mask;

        if (base == nullptr) {
            output = feat;
            o_mask = mask;
        } else {
            std::tie(output, o_mask) = (*base)(nframes, batch_size, cell_dim, var_tree->children[0], feat, mask);
        }

        auto h = autodiff::mul(output, tensor_tree::get_var(var_tree->children[1]));

        auto result = autodiff::logsoftmax(autodiff::add(
            h,
            autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[2]), h)
        ));

        return std::make_pair(result, o_mask);
    }

    res_transcriber::res_transcriber(std::shared_ptr<transcriber> base)
        : base(base)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    res_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> o_mask;
        std::tie(output, o_mask) = (*base)(nframes, batch_size, cell_dim, var_tree, feat, mask);

        return std::make_pair(autodiff::add(output, feat), o_mask);
    }

    subsampled_transcriber::subsampled_transcriber(
        int freq, int shift, std::shared_ptr<transcriber> base)
        : freq(freq), shift(shift), base(base)
    {}

    std::pair<std::shared_ptr<autodiff::op_t>,
        std::shared_ptr<autodiff::op_t>>
    subsampled_transcriber::operator()(
        int nframes,
        int batch_size,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> const& feat,
        std::shared_ptr<autodiff::op_t> const& mask) const
    {
        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> o_mask;
        std::tie(output, o_mask) = (*base)(nframes, batch_size, cell_dim, var_tree, feat, mask);

        std::vector<std::shared_ptr<autodiff::op_t>> output_vars = split_rows(output);
        std::vector<std::shared_ptr<autodiff::op_t>> o_mask_vars;
        if (o_mask != nullptr) {
            o_mask_vars = split_rows(o_mask);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> result;
        std::vector<std::shared_ptr<autodiff::op_t>> subsampled_mask;

        for (int i = 0; i < output_vars.size(); ++i) {
            if ((i - shift) % freq == 0) {
                result.push_back(output_vars[i]);

                if (o_mask != nullptr) {
                    subsampled_mask.push_back(o_mask_vars[i]);
                }
            }
        }

        if (o_mask == nullptr) {
            return std::make_pair(autodiff::row_cat(result), o_mask);
        } else {
            return std::make_pair(autodiff::row_cat(result), autodiff::row_cat(subsampled_mask));
        }
    }

}
