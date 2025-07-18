#pragma once

#include <vector>
#include <functional>

class Matrix
{
public:
    size_t rows, cols;
    std::vector<std::vector<double>> data;

    Matrix(size_t rows, size_t cols, bool randomise = false);

    static Matrix multiply(const Matrix &a, const Matrix &b);                         // Matrix * Matrix
    static Matrix transpose(const Matrix &m);                                         // Transpose
    static Matrix hadamard(const Matrix &a, const Matrix &b);                         // Element-wise *
    static Matrix applyFunction(const Matrix &m, std::function<double(double)> func); // Element-wise apply
    static Matrix subtract(const Matrix &a, const Matrix &b);                         // Matrix - Matrix
    static Matrix add(const Matrix &a, const Matrix &b);                              // Matrix + Matrix

    void print() const;
};
