#include "nn/seq2seq.h"
#include "nn/lstm-tensor-tree.h"
#include "nn/lstm-frame.h"

namespace seq2seq {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int encoder_layers)
    {
        tensor_tree::vertex root { "nil" };

        root.children.push_back(lstm_frame::make_tensor_tree(encoder_layers));

        root.children.push_back(lstm::lstm_tensor_tree_factory()());

        root.children.push_back(tensor_tree::make_tensor("label embedding"));
        root.children.push_back(tensor_tree::make_tensor("pred softmax mul"));
        root.children.push_back(tensor_tree::make_tensor("pred softmax bias"));
        root.children.push_back(tensor_tree::make_tensor("initial cell"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    seq2seq_nn_t make_training_nn(
        std::vector<int> labels,
        int label_set_size,
        std::shared_ptr<autodiff::op_t> hidden,
        int nhidden,
        int cell_dim,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        auto& comp_graph = *hidden->graph;
    
        std::shared_ptr<autodiff::op_t> cell = tensor_tree::get_var(var_tree->children[5]);
        std::shared_ptr<autodiff::op_t> output = autodiff::tanh(cell);
    
        la::cpu::tensor<double> pred_storage_t;
        pred_storage_t.resize(std::vector<unsigned int> { (unsigned int) labels.size(),
            (unsigned int) label_set_size });
        auto pred_storage = comp_graph.var(pred_storage_t);

        seq2seq_nn_t result;
    
        for (int i = 0; i < labels.size(); ++i) {
            // c = attent(output, hidden, nhidden);
            auto output_col = autodiff::weak_var(output, 0,
                std::vector<unsigned int> { (unsigned int) cell_dim, 1 });
            auto attention = autodiff::weak_var(autodiff::mul(hidden, output_col), 0,
                std::vector<unsigned int> { (unsigned int) nhidden });
            auto c = autodiff::mul(attention, hidden);

            std::shared_ptr<autodiff::op_t> pred_embedding = c;
    
            // TODO: inefficient
            auto pred = autodiff::add(autodiff::row_at(pred_storage, i),
                std::vector<std::shared_ptr<autodiff::op_t>> {
                    autodiff::logsoftmax(autodiff::add(tensor_tree::get_var(var_tree->children[4]),
                        autodiff::mul(pred_embedding, tensor_tree::get_var(var_tree->children[3]))))
                });
    
            result.preds.push_back(pred);
    
            auto input = autodiff::weak_var(autodiff::row_at(
                tensor_tree::get_var(var_tree->children[2]), labels[i]),
                0, std::vector<unsigned int> {1, (unsigned int) cell_dim});
    
            input = autodiff::add(
                autodiff::mul(input, tensor_tree::get_var(var_tree->children[1]->children[0])),
                tensor_tree::get_var(var_tree->children[1]->children[1]));
    
            la::cpu::tensor<double> output_storage_t;
            output_storage_t.resize({(unsigned int) cell_dim});
            auto output_storage = comp_graph.var(output_storage_t);
    
            lstm::lstm_step_nn_t lstm_step = lstm::make_lstm_step_nn(input, output, cell,
                tensor_tree::get_var(var_tree->children[1]->children[2]),
                tensor_tree::get_var(var_tree->children[1]->children[3]),
                tensor_tree::get_var(var_tree->children[1]->children[4]),
                tensor_tree::get_var(var_tree->children[1]->children[5]),
                nullptr,
                output_storage,
                1,
                cell_dim);
    
            output = lstm_step.output;
            cell = lstm_step.cell;
        }

        result.pred = autodiff::weak_cat(result.preds, pred_storage);

        return result;
    }

}

