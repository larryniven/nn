#include "nn/lstm-frame.h"
#include "nn/lstm.h"
#include "nn/nn.h"
#include "nn/lstm-tensor-tree.h"

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        tensor_tree::vertex result { "nil" };

        lstm::multilayer_lstm_tensor_tree_factory factory {
            std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
            lstm::bi_lstm_tensor_tree_factory {
                std::make_shared<lstm::lstm_tensor_tree_factory>(
                    lstm::lstm_tensor_tree_factory{})
            }),
            layer
        };

        result.children.push_back(factory());

        tensor_tree::vertex fc { "nil" };
        fc.children.push_back(tensor_tree::make_tensor("softmax weight"));
        fc.children.push_back(tensor_tree::make_tensor("softmax bias"));

        result.children.push_back(std::make_shared<tensor_tree::vertex>(fc));

        return std::make_shared<tensor_tree::vertex>(result);
    }

    std::shared_ptr<tensor_tree::vertex> make_dyer_tensor_tree(int layer)
    {
        tensor_tree::vertex result { "nil" };

        lstm::multilayer_lstm_tensor_tree_factory factory {
            std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
            lstm::bi_lstm_tensor_tree_factory {
                std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                    lstm::dyer_lstm_tensor_tree_factory{})
            }),
            layer
        };

        result.children.push_back(factory());

        tensor_tree::vertex fc { "nil" };
        fc.children.push_back(tensor_tree::make_tensor("softmax weight"));
        fc.children.push_back(tensor_tree::make_tensor("softmax bias"));

        result.children.push_back(std::make_shared<tensor_tree::vertex>(fc));

        return std::make_shared<tensor_tree::vertex>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_transcriber(
        std::shared_ptr<tensor_tree::vertex> param,
        double dropout,
        std::default_random_engine *gen,
        bool pyramid)
    {
        lstm::layered_transcriber result;

        int layer = param->children[0]->children.size();

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> f_trans;
            std::shared_ptr<lstm::transcriber> b_trans;

            f_trans = std::make_shared<lstm::lstm_transcriber>(
                lstm::lstm_transcriber { (int) tensor_tree::get_tensor(param->children[0]
                    ->children[i]->children[0]->children[2]).size(0) });
            b_trans = std::make_shared<lstm::lstm_transcriber>(
                lstm::lstm_transcriber { (int) tensor_tree::get_tensor(param->children[0]
                    ->children[i]->children[1]->children[2]).size(0), true });

            if (dropout != 0.0) {
                f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { f_trans, dropout, *gen });
                f_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { f_trans, dropout, *gen });

                b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { b_trans, dropout, *gen });
                b_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { b_trans, dropout, *gen });
            }

            std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { (int) tensor_tree::get_tensor(param->children[0]
                    ->children[i]->children[2]).size(1), f_trans, b_trans });

            result.layer.push_back(trans);

            if (pyramid && i != layer - 1) {
                result.layer.push_back(std::make_shared<lstm::subsampled_transcriber>(
                    lstm::subsampled_transcriber { 2, 0 }));
            }
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_dyer_transcriber(
        std::shared_ptr<tensor_tree::vertex> param,
        double dropout,
        std::default_random_engine *gen,
        bool pyramid)
    {
        lstm::layered_transcriber result;

        int layer = param->children[0]->children.size();

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> f_trans;
            std::shared_ptr<lstm::transcriber> b_trans;

            f_trans = std::make_shared<lstm::dyer_lstm_transcriber>(
                lstm::dyer_lstm_transcriber { (int) tensor_tree::get_tensor(param->children[0]
                    ->children[i]->children[0]->children[2]).size(0) });
            b_trans = std::make_shared<lstm::dyer_lstm_transcriber>(
                lstm::dyer_lstm_transcriber { (int) tensor_tree::get_tensor(param->children[0]
                    ->children[i]->children[1]->children[2]).size(0), true });

            if (dropout != 0.0) {
                f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { f_trans, dropout, *gen });
                f_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { f_trans, dropout, *gen });

                b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { b_trans, dropout, *gen });
                b_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { b_trans, dropout, *gen });
            }

            std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { (int) tensor_tree::get_tensor(param->children[0]
                    ->children[i]->children[2]).size(1), f_trans, b_trans });

            result.layer.push_back(trans);

            if (pyramid && i != layer - 1) {
                result.layer.push_back(std::make_shared<lstm::subsampled_transcriber>(
                    lstm::subsampled_transcriber { 2, 0 }));
            }
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

}
