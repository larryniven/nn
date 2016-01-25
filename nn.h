#ifndef NN_H
#define NN_H

#include "autodiff/autodiff.h"
#include "la/la.h"

namespace nn {

    struct nn_t {
        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> weight;
        std::vector<std::shared_ptr<autodiff::op_t>> bias;
        std::shared_ptr<autodiff::op_t> label_weight;
        std::shared_ptr<autodiff::op_t> label_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> hidden;
        std::shared_ptr<autodiff::op_t> output;
    };
    
    struct param_t {
        std::vector<la::matrix<double>> weight;
        std::vector<la::vector<double>> bias;
        la::matrix<double> label_weight;
        la::vector<double> label_bias;
    };

    param_t load_param(std::istream& is);
    param_t load_param(std::string filename);

    void save_param(param_t const& p, std::ostream& os);
    void save_param(param_t const& p, std::string filename);

    void iadd(param_t& p, param_t const& q);
    void resize_as(param_t& p, param_t const& q);

    struct opt_t {
        int time;
        param_t first_moment;
        param_t second_moment;
    };

    opt_t load_opt(std::istream& is);
    opt_t load_opt(std::string filename);

    void save_opt(opt_t const& o, std::ostream& os);
    void save_opt(opt_t const& o, std::string filename);

    std::pair<
        std::unordered_map<std::string, int>,
        std::vector<std::string>
    >
    load_label_map(std::string filename);

    nn_t make_nn(param_t const& p);
    nn_t make_nn2(param_t const& p);

    void adagrad_update(param_t& p, param_t const& grad, opt_t& opt_data, double step_size);
    void adam_update(param_t& p, param_t const& grad, opt_t& opt_data, double step_size);

    param_t copy_grad(nn_t const& nn);

    struct log_loss {
    
        la::vector<double> pred;
        la::vector<double> gold;
    
        log_loss(la::vector<double> const& pred,
            la::vector<double> const& gold);
    
        double loss();
    
        la::vector<double> grad();
    
    };

}

#endif
