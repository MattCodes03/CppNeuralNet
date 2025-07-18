#pragma once

#include <functional>
#include <cmath>
#include "Matrix/Matrix.h"

class Activation
{
public:
    using Func = std::function<double(double)>;

    inline static Matrix apply(const Matrix &m, Func func)
    {
        return Matrix::applyFunction(m, func);
    }

    inline static double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline static double relu(double x)
    {
        return x > 0 ? x : 0;
    }

    inline static double tanh(double x)
    {
        return std::tanh(x);
    }

    inline static double sigmoid_derivative(double x)
    {
        return x * (1.0 - x); // x assumed to be sigmoid output
    }

    inline static double relu_derivative(double x)
    {
        return x > 0.0 ? 1.0 : 0.0;
    }

    inline static double tanh_derivative(double x)
    {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
};
