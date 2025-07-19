#pragma once
#include "Interfaces/ILayer.h"
#include "Activation/Activation.h"

class Dense : public ILayer
{
public:
    Dense(size_t inputSize, size_t outputSize, ActivationFunction activation);
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &d_out) override;
    void update_weights(float learning_rate) override;

private:
    Matrix weights;
    Matrix biases;

    Matrix d_weights; // Gradient of weights
    Matrix d_biases;  // Gradient of biases

    ActivationFunction activationFunction; // Activation function

    Matrix lastInput;       // Store last input for backpropagation
    Matrix activatedOutput; // Store last activated output for backpropagation
};