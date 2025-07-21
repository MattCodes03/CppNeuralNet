#pragma once

#include "Tensor/Tensor.h"
#include <functional>

struct LossFunction
{
    using Func = std::function<double(const Tensor &, const Tensor &)>;
    using DerivativeFunc = std::function<Tensor(const Tensor &, const Tensor &)>;

    Func loss;
    DerivativeFunc derivative;
};

class Loss
{
public:
    inline static const LossFunction MSE = {
        // MSE Loss function
        [](const Tensor &predicted, const Tensor &target) -> double
        {
            Tensor diff = Tensor::subtract(predicted, target); // predicted - target

            Tensor sqr = Tensor::applyFunction(diff, [](double x)
                                               { return x * x; });

            // Sum returns double (you'll need to implement this for Tensor)
            return Tensor::sum(sqr) / (sqr.depth * sqr.rows * sqr.cols);
        },

        // Derivative of MSE w.r.t predicted output
        [](const Tensor &predicted, const Tensor &target) -> Tensor
        {
            Tensor diff = Tensor::subtract(predicted, target);

            float scalar = 2.0f / (predicted.depth * predicted.rows * predicted.cols);

            return Tensor::scale(diff, scalar);
        }};
};
