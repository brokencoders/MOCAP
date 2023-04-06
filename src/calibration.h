#pragma once 

#include <tuple>
#include <vector>

#define ALGEBRA_DEFAULT_NAMES
#include "algbera.h"

std::tuple<Algebra::Mat3x3, Algebra::Mat3x3> normalize(std::vector<Algebra::Vec2> X)
{
    using namespace Algebra;
    
    double s = 0;
    Vec2 x_bar = { 0, 0 };

    for(auto& x : X)
        x_bar += x;
        
    x_bar /= X.size();

    for(auto& x : X)
        s += (x - x_bar).norm();

    s = sqrt(2) * X.size() / s;

    for(auto& x : X)
        x = (x - x_bar) * s;

    return {{
             s, 0, -s * x_bar[0], 
             0, s, -s * x_bar[1], 
             0, 0, 1
            },
            {
             1/s,   0, x_bar[0], 
             0,   1/s, x_bar[1], 
             0,     0,        1
            }};

}