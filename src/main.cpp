#include <iostream>
#include "Matrix/Matrix.h"
#include "Activation/Activation.h"
#include "Layers/Dense.h"
#include "Net/NeuralNetwork.h"
#include "Net/Loss.h"

int main()
{
    NeuralNetwork nn;
    nn.setLossFunction(Loss::MSE);

    // Simple feedforward NN
    nn.addLayer(std::make_shared<Dense>(3, 4, Activation::RELU));
    nn.addLayer(std::make_shared<Dense>(4, 1, Activation::SIGMOID));

    // Fixed input and target
    Matrix input(3, 1);
    input.data = {
        {0.5},
        {0.1},
        {0.4}};

    Matrix target(1, 1);
    target.data = {
        {1.0}};

    std::cout << "Initial Output:\n";
    Matrix initial = nn.forward(input);
    initial.print();

    // Train over multiple epochs
    const int epochs = 1000;
    const float lr = 0.1f;

    for (int i = 0; i < epochs; ++i)
    {
        nn.train(input, target, lr);

        if (i % 100 == 0)
        {
            double loss = nn.evaluate(input, target);
            std::cout << "Epoch " << i << ", Loss: " << loss << "\n";
        }
    }

    std::cout << "\nFinal Output after training:\n";
    Matrix result = nn.forward(input);
    result.print();

    return 0;
}
