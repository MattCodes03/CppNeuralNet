#pragma once

#include "Matrix/Matrix.h"

class ILayer
{
public:
    virtual Matrix forward(const Matrix &input) = 0;
    virtual ~ILayer() = default;
};