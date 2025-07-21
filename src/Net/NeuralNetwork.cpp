#include "NeuralNetwork.h"

void NeuralNetwork::addLayer(std::shared_ptr<ILayer> layer)
{
    layers.push_back(layer);
}

Tensor NeuralNetwork::forward(const Tensor &input)
{
    Tensor output = input;
    for (const auto &layer : layers)
    {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Tensor &loss_gradient)
{
    Tensor gradient = loss_gradient;

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

void NeuralNetwork::train(const Tensor &input, const Tensor &target, float learning_rate)
{
    Tensor prediction = forward(input);

    Tensor loss_grad = lossFunction.derivative(prediction, target);

    backward(loss_grad);

    update_weights(learning_rate);
}

double NeuralNetwork::evaluate(const Tensor &input, const Tensor &target)
{
    Tensor prediction = forward(input);
    return lossFunction.loss(prediction, target);
}

void NeuralNetwork::setLossFunction(const LossFunction &lossFn)
{

    lossFunction = lossFn;
}
