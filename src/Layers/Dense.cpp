#include "Dense.h"

Dense::Dense(size_t inputSize, size_t outputSize, ActivationFunction activation)
    : weights(Tensor::fromMatrix(Matrix(outputSize, inputSize, true))),
      biases(Tensor::fromMatrix(Matrix(outputSize, 1, true))),
      activationFunction(activation)
{
}

Tensor Dense::forward(const Tensor &input)
{
    lastInput = input;

    Matrix z = Matrix::add(Matrix::multiply(weights.toMatrix(), input.toMatrix()), biases.toMatrix());

    // Activation is element-wise
    Matrix activated = activationFunction.apply(z);

    activatedOutput = Tensor::fromMatrix(activated);
    return activatedOutput;
}

Tensor Dense::backward(const Tensor &d_out)
{
    Matrix d_activation = Matrix::hadamard(
        d_out.toMatrix(),
        activationFunction.applyDerivative(activatedOutput.toMatrix()));

    // Gradients
    d_weights = Tensor::fromMatrix(Matrix::multiply(
        d_activation,
        Matrix::transpose(lastInput.toMatrix())));

    d_biases = Tensor::fromMatrix(d_activation);

    // Return grad to propagate to previous layer
    Matrix grad_input = Matrix::multiply(Matrix::transpose(weights.toMatrix()), d_activation);
    return Tensor::fromMatrix(grad_input);
}

void Dense::update_weights(float learning_rate)
{
    weights = Tensor::subtract(weights, Tensor::scale(d_weights, learning_rate));
    biases = Tensor::subtract(biases, Tensor::scale(d_biases, learning_rate));
}
