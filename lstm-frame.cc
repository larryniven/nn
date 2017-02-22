#include "nn/lstm-frame.h"
#include "nn/lstm.h"
#include "nn/nn.h"
#include "nn/lstm-tensor-tree.h"

namespace lstm_frame {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
    {
        tensor_tree::vertex result { tensor_tree::tensor_t::nil };

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

    std::shared_ptr<tensor_tree::vertex> make_hypercolumn_tensor_tree(int layer)
    {
        tensor_tree::vertex result { tensor_tree::tensor_t::nil };

        lstm::multilayer_lstm_tensor_tree_factory factory {
            std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
            lstm::bi_lstm_tensor_tree_factory {
                std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                    lstm::dyer_lstm_tensor_tree_factory{})
            }),
            layer
        };

        tensor_tree::vertex hypercolumn { tensor_tree::tensor_t::nil };

        hypercolumn.children.push_back(factory());

        for (int i = 0; i < layer + 1; ++i) {
            hypercolumn.children.push_back(tensor_tree::make_tensor("hypercolumn weight"));
        }
        hypercolumn.children.push_back(tensor_tree::make_tensor("hypercolumn bias"));

        result.children.push_back(std::make_shared<tensor_tree::vertex>(hypercolumn));

        result.children.push_back(tensor_tree::make_tensor("softmax weight"));
        result.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_pyramid_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::step_transcriber> step;

        if (dropout != 0.0) {
            assert(gen != nullptr);

            step = std::make_shared<lstm::input_dropout_transcriber>(
                lstm::input_dropout_transcriber {
                    *gen, dropout,
                    std::make_shared<lstm::dyer_lstm_step_transcriber>(
                    lstm::dyer_lstm_step_transcriber{})
                });
        } else {
            step = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});
        }

        lstm::layered_transcriber result;

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> trans;

            if (dropout != 0.0) {
                trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber {
                        std::make_shared<lstm::output_dropout_transcriber>(
                        lstm::output_dropout_transcriber {
                            *gen, dropout, step })
                    });
            } else {
                trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
            }

            trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { trans });

            if (i != layer - 1) {
                trans = std::make_shared<lstm::subsampled_transcriber>(
                    lstm::subsampled_transcriber { 2, 0, trans });
            }

            result.layer.push_back(trans);
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_hypercolumn_transcriber(
        int layer,
        double dropout,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::step_transcriber> step;

        if (dropout != 0.0) {
            assert(gen != nullptr);

            step = std::make_shared<lstm::input_dropout_transcriber>(
                lstm::input_dropout_transcriber {
                    *gen, dropout,
                    std::make_shared<lstm::dyer_lstm_step_transcriber>(
                    lstm::dyer_lstm_step_transcriber{})
                });
        } else {
            step = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});
        }

        lstm::hypercolumn_transcriber result;

        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<lstm::transcriber> trans;

            if (dropout != 0.0) {
                trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber {
                        std::make_shared<lstm::output_dropout_transcriber>(
                        lstm::output_dropout_transcriber {
                            *gen, dropout, step })
                    });
            } else {
                trans = std::make_shared<lstm::lstm_transcriber>(
                    lstm::lstm_transcriber { step });
            }

            trans = std::make_shared<lstm::bi_transcriber>(
                lstm::bi_transcriber { trans });

            if (i != layer - 1) {
                trans = std::make_shared<lstm::subsampled_transcriber>(
                    lstm::subsampled_transcriber { 2, 0, trans });
            }

            result.layer.push_back(trans);
        }

        return std::make_shared<lstm::hypercolumn_transcriber>(result);
    }

}
