#include "nn/nn-gpu.h"
#include "opt/opt-gpu.h"

namespace nn {

    namespace gpu {

        l2_loss::l2_loss(la::gpu::tensor_like<double> const& gold, la::gpu::tensor_like<double> const& pred)
            : gold(gold), pred(pred)
        {}
        
        double l2_loss::loss()
        {
            la::gpu::tensor<double> diff;
            diff.resize(gold.sizes());
            la::gpu::copy(diff, gold);
            la::gpu::axpy(diff, -1, pred);

            return la::gpu::dot(diff, diff);
        }
        
        la::gpu::tensor<double> l2_loss::grad(double scale)
        {
            la::gpu::tensor<double> g;
            g.resize(pred.sizes());
            la::gpu::axpy(g, 2 * scale, pred);
            la::gpu::axpy(g, -2 * scale, gold);

            return g;
        }

        log_loss::log_loss(la::gpu::tensor_like<double> const& gold,
            la::gpu::tensor_like<double> const& pred)
            : gold(gold), pred(pred)
        {}
        
        double log_loss::loss()
        {
            return -la::gpu::dot(pred, gold);
        }
        
        la::gpu::tensor<double> log_loss::grad(double scale)
        {
            la::gpu::tensor<double> result;
            la::gpu::resize_as(result, gold);
            la::gpu::axpy(result, -scale, gold);
            return result;
        }
    }
}
