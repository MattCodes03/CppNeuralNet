#include "Dense.h"

Dense::Dense(size_t inputSize, size_t outputSize, ActivationFunction activation) : weights(outputSize, inputSize, true),
                                                                                   biases(outputSize, 1, true),
                                                                                   activationFunction(activation)
{
}

Matrix Dense::forward(const Matrix &input)
{
    lastInput = input; // Save input for weight gradient calc

    // Perform the forward pass: output = weights * input + biases
    Matrix z = Matrix::add(Matrix::multiply(weights, input), biases);

    // Apply the activation function
    activatedOutput = activationFunction.apply(z);
    return activatedOutput;
}

Matrix Dense::backward(const Matrix &d_out)
{
    Matrix d_activation = Matrix::hadamard(d_out, activationFunction.applyDerivative(activatedOutput));

    d_weights = Matrix::multiply(d_activation, Matrix::transpose(lastInput));

    d_biases = d_activation; // Biases are simply the derivative of the output

    return Matrix::multiply(Matrix::transpose(weights), d_activation);
}

void Dense::update_weights(float learning_rate)
{
    weights = Matrix::subtract(weights, Matrix::multiply(d_weights, learning_rate));
    biases = Matrix::subtract(biases, Matrix::multiply(d_biases, learning_rate));
}
