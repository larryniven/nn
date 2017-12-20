#include "nn/cnn.h"

namespace cnn {

    transcriber::~transcriber()
    {}

    bool transcriber::require_param() const
    {
        return true;
    }

    conv_transcriber::conv_transcriber()
        : p1(0), p2(0), d1(1), d2(1)
    {}

    conv_transcriber::conv_transcriber(int p1, int p2, int d1, int d2)
        : p1(p1), p2(p2), d1(d1), d2(d2)
    {}

    std::shared_ptr<autodiff::op_t>
    conv_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        auto k = autodiff::corr_linearize(input,
            tensor_tree::get_var(var_tree->children[0]), p1, p2, d1, d2);

        auto z = autodiff::mul(k, tensor_tree::get_var(var_tree->children[0]));

        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), z);

        return autodiff::relu(autodiff::add(z, b));
    }

    max_pooling_transcriber::max_pooling_transcriber(
            int dim1, int dim2)
        : dim1(dim1), dim2(dim2), stride1(dim1), stride2(dim2)
    {}

    max_pooling_transcriber::max_pooling_transcriber(
            int dim1, int dim2, int stride1, int stride2)
        : dim1(dim1), dim2(dim2), stride1(stride1), stride2(stride2)
    {}

    bool max_pooling_transcriber::require_param() const
    {
        return false;
    }

    std::shared_ptr<autodiff::op_t>
    max_pooling_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        return autodiff::pool_max(input, dim1, dim2, stride1, stride2);
    }

    std::shared_ptr<autodiff::op_t>
    fc_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(input);
        std::vector<unsigned int> sizes = t.sizes();
        unsigned int dim = 1;
        while (sizes.size() > 1) {
            dim *= sizes.back();
            sizes.pop_back();
        }
        sizes.push_back(dim);
        auto v = autodiff::reshape(input, sizes);

        auto z = autodiff::mul(v, tensor_tree::get_var(var_tree->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), z);
        return autodiff::add(z, b);
    }

    std::shared_ptr<autodiff::op_t>
    framewise_fc_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        auto& t = autodiff::get_output<la::cpu::tensor<double>>(input);
        auto m = autodiff::reshape(input, {t.size(0), t.vec_size() / t.size(0)});
        auto z = autodiff::mul(m, tensor_tree::get_var(var_tree->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[1]), z);

        return autodiff::add(z, b);
    }

    bool relu_transcriber::require_param() const
    {
        return false;
    }

    std::shared_ptr<autodiff::op_t>
    relu_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        return autodiff::relu(input);
    }

    dropout_transcriber::dropout_transcriber(double prob, std::default_random_engine& gen)
        : prob(prob), gen(gen)
    {}

    bool dropout_transcriber::require_param() const
    {
        return false;
    }

    std::shared_ptr<autodiff::op_t>
    dropout_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        return autodiff::emul(input, autodiff::dropout_mask(input, prob, gen));
    }

    std::shared_ptr<autodiff::op_t>
    multilayer_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        std::shared_ptr<autodiff::op_t> feat = input;

        int j = 0;

        for (int i = 0; i < layers.size(); ++i) {
            if (layers[i]->require_param()) {
                feat = (*layers[i])(var_tree->children[j], feat);
                ++j;
            } else {
                feat = (*layers[i])(nullptr, feat);
            }
        }

        return feat;
    }

    bool logsoftmax_transcriber::require_param() const
    {
        return false;
    }

    std::shared_ptr<autodiff::op_t>
    logsoftmax_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        return autodiff::logsoftmax(input);
    }

    densenet_transcriber::densenet_transcriber(int transcriber)
        : transcriber(transcriber)
    {}

    std::shared_ptr<autodiff::op_t>
    densenet_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> transcribers;

        int ell = 0;

        std::shared_ptr<autodiff::op_t> feat = input;

        for (int i = 0; i < transcriber; ++i) {
            auto t = autodiff::corr_linearize(feat,
                tensor_tree::get_var(var_tree->children[ell + i]));
            transcribers.push_back(t);

            std::vector<std::shared_ptr<autodiff::op_t>> muls;
            for (int k = 0; k < i + 1; ++k) {
                muls.push_back(autodiff::mul(tensor_tree::get_var(var_tree->children[ell + k]),
                    transcribers[k]));
            }
            feat = autodiff::relu(autodiff::add(muls));

            ell += i + 1;
        }

        return input;
    }

    std::vector<std::shared_ptr<autodiff::op_t>> ifo_pooling(
        std::vector<std::shared_ptr<autodiff::op_t>> input,
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate,
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate,
        std::vector<std::shared_ptr<autodiff::op_t>> output_gate)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> result;
        std::shared_ptr<autodiff::op_t> cell = nullptr;

        for (int i = 0; i < input.size(); ++i) {
            if (cell == nullptr) {
                cell = autodiff::emul(input[i], input_gate[i]);
            } else {
                cell = autodiff::add(autodiff::emul(input[i], input_gate[i]),
                    autodiff::emul(cell, forget_gate[i]));
            }

            result.push_back(autodiff::emul(cell, output_gate[i]));
        }

        return result;
    }

    std::vector<std::shared_ptr<autodiff::op_t>> conv_ifo_pooling(
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> input_gate,
        std::shared_ptr<autodiff::op_t> forget_gate,
        std::shared_ptr<autodiff::op_t> output_gate,
        int size)
    {
        std::vector<std::shared_ptr<autodiff::op_t>> output_vec;
        std::vector<std::shared_ptr<autodiff::op_t>> input_gate_vec;
        std::vector<std::shared_ptr<autodiff::op_t>> forget_gate_vec;
        std::vector<std::shared_ptr<autodiff::op_t>> output_gate_vec;

        for (int i = 0; i < size; ++i) {
            output_vec.push_back(autodiff::row_at(output, i));
            input_gate_vec.push_back(autodiff::row_at(input_gate, i));
            forget_gate_vec.push_back(autodiff::row_at(forget_gate, i));
            output_gate_vec.push_back(autodiff::row_at(output_gate, i));
        }

        return ifo_pooling(output_vec, input_gate_vec, forget_gate_vec, output_gate_vec);
    }

    std::vector<std::shared_ptr<autodiff::op_t>> conv_fo_pooling(
        std::shared_ptr<autodiff::op_t> output,
        std::shared_ptr<autodiff::op_t> forget_gate,
        std::shared_ptr<autodiff::op_t> output_gate,
        int size)
    {
        auto one = autodiff::resize_as(forget_gate, 1);
        auto input_gate = autodiff::sub(one, forget_gate);

        return conv_ifo_pooling(output, input_gate, forget_gate, output_gate, size);
    }

    conv_fo_pooling_transcriber::conv_fo_pooling_transcriber(
        unsigned int rows, unsigned int cols,
        std::shared_ptr<transcriber> input_conv,
        std::shared_ptr<transcriber> forget_gate_conv,
        std::shared_ptr<transcriber> output_gate_conv)
        : rows(rows), cols(cols)
        , input_conv(input_conv), forget_gate_conv(forget_gate_conv)
        , output_gate_conv(output_gate_conv)
    {}

    std::shared_ptr<autodiff::op_t>
    conv_fo_pooling_transcriber::operator()(std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> input)
    {
        auto input_lin = autodiff::corr_linearize(input,
            tensor_tree::get_var(var_tree->children[0]->children[0]));

        auto input_res = autodiff::mul(input_lin,
            tensor_tree::get_var(var_tree->children[0]->children[0]));
        auto input_bias = autodiff::rep_row_to(
            tensor_tree::get_var(var_tree->children[0]->children[1]), input_res);
        input_res = autodiff::add(input_res, input_bias);

        auto forget_res = autodiff::mul(input_lin,
            tensor_tree::get_var(var_tree->children[1]->children[0]));
        auto forget_bias = autodiff::rep_row_to(
            tensor_tree::get_var(var_tree->children[1]->children[1]), forget_res);
        forget_res = autodiff::add(forget_res, forget_bias);

        auto output_res = autodiff::mul(input_lin,
            tensor_tree::get_var(var_tree->children[1]->children[0]));
        auto output_bias = autodiff::rep_row_to(
            tensor_tree::get_var(var_tree->children[1]->children[1]), output_res);
        output_res = autodiff::add(output_res, output_bias);

        auto& t = autodiff::get_output<la::tensor_like<double>>(tensor_tree::get_var(
            var_tree->children[0]->children[1]));

        input_res = autodiff::reshape(input_res, { rows, cols * t.size(0) });
        forget_res = autodiff::reshape(forget_res, { rows, cols * t.size(0) });
        output_res = autodiff::reshape(output_res, { rows, cols * t.size(0) });

        return autodiff::reshape(autodiff::row_cat(conv_fo_pooling(
            input_res, forget_res, output_res, rows)), { rows, cols, t.size(0) });
    }

}
