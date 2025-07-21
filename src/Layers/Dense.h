#pragma once
#include "Interfaces/ILayer.h"
#include "Activation/Activation.h"

class Dense : public ILayer
{
public:
    Dense(size_t inputSize, size_t outputSize, ActivationFunction activation);
    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &d_out) override;
    void update_weights(float learning_rate) override;

private:
    Tensor weights;
    Tensor biases;

    Tensor d_weights; // Gradient of weights
    Tensor d_biases;  // Gradient of biases

    ActivationFunction activationFunction; // Activation function

    Tensor lastInput;       // Store last input for backpropagation
    Tensor activatedOutput; // Store last activated output for backpropagation
};