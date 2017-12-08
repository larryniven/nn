#ifndef NN_GPU_H
#define NN_GPU_H

#include "nn/nn.h"
#include "la/la-gpu.h"

namespace nn {

    namespace gpu {

        struct l2_loss {

            la::gpu::tensor_like<double> const& gold;
            la::gpu::tensor_like<double> const& pred;
        
            l2_loss(la::gpu::tensor_like<double> const& gold,
                la::gpu::tensor_like<double> const& pred);
        
            double loss();
        
            la::gpu::tensor<double> grad(double scale=1);
        
        };

        struct log_loss {
        
            la::gpu::tensor_like<double> const& pred;
            la::gpu::tensor_like<double> const& gold;
        
            log_loss(la::gpu::tensor_like<double> const& gold,
                la::gpu::tensor_like<double> const& pred);
        
            double loss();
        
            la::gpu::tensor<double> grad(double scale = 1);
        
        };
    }
}

#endif
