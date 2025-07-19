#pragma once

#include <vector>
#include <memory>
#include "Matrix/Matrix.h"
#include "Interfaces/ILayer.h"
#include "Loss.h"

class NeuralNetwork
{
public:
    NeuralNetwork() = default;

    void addLayer(std::shared_ptr<ILayer> layer);
    void setLossFunction(LossFunction lf) { lossFunction = lf; }

    Matrix forward(const Matrix &input);

    void backward(const Matrix &loss_gradient);
    void update_weights(float learning_rate);
    void train(const Matrix &input, const Matrix &target, float learning_rate);
    double evaluate(const Matrix &input, const Matrix &target);

private:
    std::vector<std::shared_ptr<ILayer>> layers; // Store layers in the network
    LossFunction lossFunction;
};