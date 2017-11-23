#include "nn/nn-gpu.h"
#include "opt/opt-gpu.h"

namespace nn {

    namespace gpu {

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
