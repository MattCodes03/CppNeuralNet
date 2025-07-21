#pragma once
#include <iostream>
#include <vector>
#include "Matrix/Matrix.h"

class Tensor
{
public:
    Tensor() = default;

    size_t depth, rows, cols;
    std::vector<Matrix> slices;

    Tensor(size_t depth, size_t rows, size_t cols, bool randomise = false)
        : depth(depth), rows(rows), cols(cols)
    {
        for (size_t i = 0; i < depth; ++i)
        {
            slices.emplace_back(Matrix(rows, cols, randomise));
        }
    }

    Matrix &operator[](size_t index)
    {
        return slices.at(index);
    }

    const Matrix &operator[](size_t index) const
    {
        return slices.at(index);
    }

    static Tensor zeros(size_t depth, size_t rows, size_t cols)
    {
        return Tensor(depth, rows, cols, false);
    }

    static Tensor add(const Tensor &a, const Tensor &b)
    {
        if (a.depth != b.depth || a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument("Tensor::add: shape mismatch");

        Tensor result(a.depth, a.rows, a.cols);
        for (size_t d = 0; d < a.depth; ++d)
        {
            result[d] = Matrix::add(a[d], b[d]);
        }
        return result;
    }

    static Tensor subtract(const Tensor &a, const Tensor &b)
    {
        if (a.depth != b.depth || a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument("Tensor::subtract: shape mismatch");

        Tensor result(a.depth, a.rows, a.cols);
        for (size_t d = 0; d < a.depth; ++d)
        {
            result[d] = Matrix::subtract(a[d], b[d]);
        }
        return result;
    }

    static double sum(const Tensor &t)
    {
        double total = 0.0;
        for (size_t d = 0; d < t.depth; ++d)
        {
            total += Matrix::sum(t[d]);
        }
        return total;
    }

    static Tensor scale(const Tensor &t, float scalar)
    {
        Tensor result(t.depth, t.rows, t.cols);
        for (size_t d = 0; d < t.depth; ++d)
        {
            result[d] = Matrix::multiply(t[d], scalar);
        }
        return result;
    }

    static Tensor hadamard(const Tensor &a, const Tensor &b)
    {
        if (a.depth != b.depth || a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument("Tensor::hadamard: shape mismatch");

        Tensor result(a.depth, a.rows, a.cols);
        for (size_t d = 0; d < a.depth; ++d)
        {
            result[d] = Matrix::hadamard(a[d], b[d]);
        }
        return result;
    }

    static Tensor applyFunction(const Tensor &t, std::function<double(double)> func)
    {
        Tensor result(t.depth, t.rows, t.cols);
        for (size_t d = 0; d < t.depth; ++d)
        {
            result[d] = Matrix::applyFunction(t[d], func);
        }
        return result;
    }

    static Tensor transpose(const Tensor &t)
    {
        if (t.depth != 1)
            throw std::invalid_argument("Tensor::transpose only supports 1-slice tensors");

        Tensor result(1, t.cols, t.rows); // Swap rows and cols
        result[0] = Matrix::transpose(t[0]);
        return result;
    }

    static Tensor fromMatrix(const Matrix &m)
    {
        Tensor t(1, m.rows, m.cols);
        t[0] = m;
        return t;
    }

    Matrix toMatrix() const
    {
        if (depth != 1)
            throw std::runtime_error("Tensor::toMatrix only valid for depth == 1");

        return slices[0];
    }

    static Tensor matmul(const Tensor &a, const Tensor &b)
    {
        if (a.depth != 1 || b.depth != 1)
            throw std::invalid_argument("Tensor::matmul only supports depth == 1 tensors");

        Matrix result = Matrix::multiply(a[0], b[0]);

        return Tensor::fromMatrix(result); // wrap in Tensor
    }

    void print() const
    {
        for (size_t i = 0; i < depth; ++i)
        {
            std::cout << "Slice " << i << ":\n";
            slices[i].print();
        }
    }
};