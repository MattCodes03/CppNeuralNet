#include <iostream>
#include "Matrix/Matrix.h"
#include "Activation/Activation.h"

int main()
{
    std::cout << "Matrix A\n";
    Matrix A(2, 3, true);
    A.print();

    std::cout << "Sigmoid\n";
    Matrix C = Matrix::applyFunction(A, Activation::sigmoid);
    C.print();

    std::cout << "tanh\n";
    Matrix D = Matrix::applyFunction(A, Activation::tanh);
    D.print();

    std::cout << "ReLU\n";
    Matrix E = Matrix::applyFunction(A, Activation::relu);
    E.print();

    std::cout << "Sigmoid Derivative\n";
    Matrix F = Matrix::applyFunction(C, Activation::sigmoid_derivative);
    F.print();

    std::cout << "tanh Derivative\n";
    Matrix G = Matrix::applyFunction(A, Activation::tanh_derivative);
    G.print();

    std::cout << "ReLU Derivative\n";
    Matrix H = Matrix::applyFunction(A, Activation::relu_derivative);
    H.print();

    return 0;
}
