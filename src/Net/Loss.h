#pragma once

#include "Matrix/Matrix.h"

struct LossFunction
{
    using Func = std::function<double(const Matrix &, const Matrix &)>;
    using DerivativeFunc = std::function<Matrix(const Matrix &, const Matrix &)>;

    Func loss;
    DerivativeFunc derivative;
};

class Loss
{
public:
    inline static const LossFunction MSE = {
        // MSE Loss function
        [](const Matrix &predicted, const Matrix &target) -> double
        {
            Matrix diff = Matrix::subtract(predicted, target); // diff = predicted - target

            // Square each element: diff^2
            Matrix sqr = Matrix::applyFunction(diff, [](double x)
                                               { return x * x; });

            return Matrix::sum(sqr) / (sqr.rows * sqr.cols); // Mean of squared errors
        },

        // Derivative of MSE w.r.t predicted output
        [](const Matrix &predicted, const Matrix &target) -> Matrix
        {
            Matrix diff = Matrix::subtract(predicted, target); // diff = predicted - target

            // Multiply diff by scalar 2 / N, where N = total elements
            float scalar = 2.0f / (predicted.rows * predicted.cols);
            return Matrix::multiply(diff, scalar);
        }};
};
