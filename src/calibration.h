#pragma once 

#include <tuple>
#include <vector>

#include "algbera.h"


inline std::vector<Algebra::Vector> generateCheckerboardCoords(int layout_x, int layout_y, double size)
{
    std::vector<Algebra::Vector> checkerboard(layout_x * layout_y, Algebra::Vector(2));
    int x = -1, y = -1;

    for (auto& pt : checkerboard)
    {
        x = (x + 1) % layout_x;
        if (x == 0) y++;
        pt[0] = x * size;
        pt[1] = y * size;
    }
    return checkerboard;
}

inline std::tuple<Algebra::Matrix, Algebra::Matrix> normalize2Dset(std::vector<Algebra::Vector>& X)
{
    using namespace Algebra;
    
    double s = 0;
    Vector x_bar = { 0, 0 };

    for(auto& x : X)
        x_bar += x;
        
    x_bar /= X.size();

    for(auto& x : X)
        s += (x - x_bar).norm();

    s = sqrt(2) * X.size() / s;

    for(auto& x : X)
        x = (x - x_bar) * s;

    return {{ 3, 3, {
             s, 0, -s * x_bar[0], 
             0, s, -s * x_bar[1], 
             0, 0, 1
            }},
            { 3, 3, {
             1/s,   0, x_bar[0], 
             0,   1/s, x_bar[1], 
             0,     0,        1
            }}};

}

inline void unnormalizeSet(std::vector<Algebra::Vector>& X, const Algebra::Matrix& N_inverse)
{
    for(auto& x : X)
        x = (N_inverse * x.hom()).hom_i();
}
