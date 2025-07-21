#pragma once

#include "Tensor/Tensor.h"

class ILayer
{
public:
    virtual ~ILayer() = default;

    virtual Tensor forward(const Tensor &input) = 0;
    virtual Tensor backward(const Tensor &d_out) = 0;
    virtual void update_weights(float learning_rate) = 0;
};