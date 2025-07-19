#pragma once

#include "Matrix/Matrix.h"

class ILayer
{
public:
    virtual ~ILayer() = default;

    virtual Matrix forward(const Matrix &input) = 0;
    virtual Matrix backward(const Matrix &d_out) = 0;
    virtual void update_weights(float learning_rate) = 0;
};