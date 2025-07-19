#include "NeuralNetwork.h"

void NeuralNetwork::addLayer(std::shared_ptr<ILayer> layer)
{
    layers.push_back(layer);
}

Matrix NeuralNetwork::forward(const Matrix &input)
{
    Matrix output = input;
    for (const auto &layer : layers)
    {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Matrix &loss_gradient)
{
    Matrix gradient = loss_gradient;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
    {
        gradient = layers[i]->backward(gradient);
    }
}

void NeuralNetwork::update_weights(float learning_rate)
{
    for (auto &layer : layers)
    {
        layer->update_weights(learning_rate);
    }
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target, float learning_rate)
{
    Matrix prediction = forward(input);

    Matrix loss_grad = lossFunction.derivative(prediction, target);

    backward(loss_grad);

    update_weights(learning_rate);
}

double NeuralNetwork::evaluate(const Matrix &input, const Matrix &target)
{
    Matrix prediction = forward(input);
    return lossFunction.loss(prediction, target);
}