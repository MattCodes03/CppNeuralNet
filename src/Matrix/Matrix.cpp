#include "Matrix.h"
#include <iostream>
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols, bool randomise)
    : rows(rows), cols(cols)
{
    data.resize(rows, std::vector<double>(cols));
    if (randomise)
    {
        for (auto &row : data)
            for (auto &val : row)
                val = (double)rand() / RAND_MAX * 2 - 1; // Random between -1 and 1
    }
}

Matrix Matrix::multiply(const Matrix &a, const Matrix &b)
{
    if (a.cols != b.rows)
    {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    }

    Matrix result(a.rows, b.cols);

    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < b.cols; ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < a.cols; ++k)
            {
                sum += a.data[i][k] * b.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }

    return result;
}

Matrix Matrix::multiply(const Matrix &m, float scalar)
{
    Matrix result = m;

    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            result.data[i][j] *= scalar;
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix &m)
{
    Matrix result(m.cols, m.rows);

    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            result.data[j][i] = m.data[i][j];
        }
    }

    return result;
}

Matrix Matrix::hadamard(const Matrix &a, const Matrix &b)
{
    if (a.cols != b.cols || a.rows != b.rows)
    {
        throw std::invalid_argument("Incompatible matrix dimensions for Hadamard product (element-wise product).");
    }

    Matrix result(a.rows, a.cols);

    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < b.cols; ++j)
        {
            result.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }

    return result;
}

Matrix Matrix::applyFunction(const Matrix &m, std::function<double(double)> func)
{
    Matrix result(m.rows, m.cols);

    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            result.data[i][j] = func(m.data[i][j]);
        }
    }

    return result;
}

Matrix Matrix::add(const Matrix &a, const Matrix &b)
{
    if (a.cols != b.cols || a.rows != b.rows)
    {
        throw std::invalid_argument("Incompatible matrix dimensions for addition.");
    }

    Matrix result(a.rows, a.cols);

    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < b.cols; ++j)
        {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return result;
}

double Matrix::sum(const Matrix &m)
{
    double total = 0.0;
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            total += m.data[i][j];
        }
    }
    return total;
}

Matrix Matrix::subtract(const Matrix &a, const Matrix &b)
{
    if (a.cols != b.cols || a.rows != b.rows)
    {
        throw std::invalid_argument("Incompatible matrix dimensions for subtraction.");
    }

    Matrix result(a.rows, a.cols);

    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < b.cols; ++j)
        {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }

    return result;
}

void Matrix::print() const
{
    for (size_t i = 0; i < rows; ++i)
    {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << data[i][j];
            if (j < cols - 1)
                std::cout << ", ";
        }
        std::cout << " ]\n";
    }
    std::cout << "\n\n";
}
