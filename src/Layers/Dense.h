#pragma once
#include "Interfaces/ILayer.h"
#include "Activation/Activation.h"

class Dense : public ILayer
{
public:
    Dense(size_t inputSize, size_t outputSize, Activation::Func activation);
    Matrix forward(const Matrix &input) override;

private:
    Matrix weights;
    Matrix biases;
};