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
                std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                    lstm::dyer_lstm_tensor_tree_factory{})
            }),
            layer
        };

        result.children.push_back(factory());
        result.children.push_back(tensor_tree::make_tensor("softmax weight"));
        result.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(result);
    }

    std::shared_ptr<tensor_tree::vertex> make_uni_tensor_tree(int layer)
    {
        tensor_tree::vertex result { "nil" };

        lstm::multilayer_lstm_tensor_tree_factory factory {
            std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                lstm::dyer_lstm_tensor_tree_factory{}),
            layer
        };

        result.children.push_back(factory());
        result.children.push_back(tensor_tree::make_tensor("softmax weight"));
        result.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::step_transcriber> step
            = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});

        lstm::layered_transcriber result;

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> f_trans;
            std::shared_ptr<lstm::transcriber> b_trans;

            if (dropout != 0.0) {
                f_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { f_trans, dropout, *gen });
                f_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { f_trans, dropout, *gen });

                b_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step, true });
                b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { b_trans, dropout, *gen });
                b_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { b_trans, dropout, *gen });
            } else {
                f_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                b_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step, true });
            }

            std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { f_trans, b_trans });

            result.layer.push_back(trans);
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_res_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::step_transcriber> step
            = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});

        lstm::layered_transcriber result;

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> f_trans;
            std::shared_ptr<lstm::transcriber> b_trans;

            if (dropout != 0.0) {
                f_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { f_trans, dropout, *gen });
                f_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { f_trans, dropout, *gen });

                b_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step, true });
                b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { b_trans, dropout, *gen });
                b_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { b_trans, dropout, *gen });
            } else {
                f_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                b_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step, true });
            }

            std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { f_trans, b_trans });

            trans = std::make_shared<lstm::res_transcriber>(lstm::res_transcriber(trans));

            result.layer.push_back(trans);
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_uni_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::step_transcriber> step
            = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});

        lstm::layered_transcriber result;

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> trans;

            if (dropout != 0.0) {
                trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { trans, dropout, *gen });
                trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { trans, dropout, *gen });
            } else {
                trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
            }

            result.layer.push_back(trans);
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_pyramid_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::step_transcriber> step
            = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});

        lstm::layered_transcriber result;

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> f_trans;
            std::shared_ptr<lstm::transcriber> b_trans;

            if (dropout != 0.0) {
                f_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { f_trans, dropout, *gen });
                f_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { f_trans, dropout, *gen });

                b_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step, true });
                b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                    lstm::input_dropout_transcriber { b_trans, dropout, *gen });
                b_trans = std::make_shared<lstm::output_dropout_transcriber>(
                    lstm::output_dropout_transcriber { b_trans, dropout, *gen });
            } else {
                f_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
                b_trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step, true });
            }

            std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { f_trans, b_trans });

            if (i != layer - 1) {
                trans = std::make_shared<lstm::subsampled_transcriber>(
                    lstm::subsampled_transcriber { 2, 0, trans });
            }

            result.layer.push_back(trans);
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

}
