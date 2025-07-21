#pragma once

#include <vector>
#include <memory>
#include "Tensor/Tensor.h"
#include "Interfaces/ILayer.h"
#include "Net/Loss.h" // Assuming you have a LossFunction struct/class

class NeuralNetwork
{
public:
    NeuralNetwork() = default;

    void addLayer(std::shared_ptr<ILayer> layer);

    Tensor forward(const Tensor &input);

    void backward(const Tensor &loss_gradient);

    void update_weights(float learning_rate);

    void train(const Tensor &input, const Tensor &target, float learning_rate);

    double evaluate(const Tensor &input, const Tensor &target);

    void setLossFunction(const LossFunction &lossFn);

private:
    std::vector<std::shared_ptr<ILayer>> layers;
    LossFunction lossFunction;
};
