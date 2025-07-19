#pragma once

#include <functional>
#include <cmath>
#include "Matrix/Matrix.h"

// Struct to hold an activation function and its derivative
struct ActivationFunction
{
    using Func = std::function<double(double)>;
    Func forward;
    Func derivative;

    // Apply the forward activation to a matrix
    Matrix apply(const Matrix &m) const
    {
        return Matrix::applyFunction(m, forward);
    }

    // Apply the derivative activation to a matrix
    Matrix applyDerivative(const Matrix &m) const
    {
        return Matrix::applyFunction(m, derivative);
    }
};

class Activation
{
public:
    // Sigmoid forward
    static double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Sigmoid derivative (expects sigmoid output as input)
    static double sigmoid_derivative(double x)
    {
        return x * (1.0 - x);
    }

    // ReLU forward
    static double relu(double x)
    {
        return x > 0 ? x : 0;
    }

    // ReLU derivative
    static double relu_derivative(double x)
    {
        return x > 0.0 ? 1.0 : 0.0;
    }

    // Tanh forward
    static double tanh(double x)
    {
        return std::tanh(x);
    }

    // Tanh derivative
    static double tanh_derivative(double x)
    {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }

    inline static const ActivationFunction SIGMOID = {
        sigmoid,
        sigmoid_derivative};

    inline static const ActivationFunction RELU = {
        relu,
        relu_derivative};

    inline static const ActivationFunction TANH = {
        tanh,
        tanh_derivative};
};
