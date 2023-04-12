#pragma once 

#include <tuple>
#include <vector>
#include <functional>
#include <algorithm>

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

Algebra::Vector minimizelm(std::function<Algebra::Vector(Algebra::Vector)> f, 
        Algebra::Vector x0, std::function<Algebra::Matrix(Algebra::Vector)> jac, 
        double e1 = 1e-8, double e2 = 1e-8, double t = 0.001, int k_max = 100) 
{
    double v = 2.0;
    Algebra::Vector x = x0;
    Algebra::Matrix J = jac(x);
    Algebra::Matrix JT = J.T();
    
    Algebra::Matrix A = JT * J;
    Algebra::Vector g = JT * f(x);
    
    bool found = g.absMax() <= e1;
    double u = t * A.max();

    for (int k = 0; !found && k < k_max; k++)
    {
        Algebra::Vector h_lm = (A + u * Algebra::identity(A.rows)).solve(-g);
        if (h_lm.norm() <= e2 * (x.norm() + e2))
            found = true;
        else
        {
            Algebra::Vector x_new = x + h_lm;
            Algebra::Vector fx = f(x);
            Algebra::Vector fx_new = f(x_new);
            double o = (fx.T() * fx - fx_new.T() * fx_new) / (h_lm.T() * (u * h_lm - g));
            if (o > 0)
            {
                x = x_new;
                J = jac(x);
                JT = J.T();
                A = JT * J;
                g = JT * f(x);
                found = g.absMax() <= e1;
                u = u * std::max(1.0 / 3.0, 1.0 - std::pow(2 * o - 1, 3));
                v = 2.0;
            } 
            else
            {
                u = u * v;
                v *= 2.0;
            }
        }
    }
    return x;
}

Algebra::Vector multiPointProjectionError(const std::vector<Algebra::Vector>& X,
                                          Algebra::Vector h,
                                          const Algebra::Vector& obs_prj_pts)
{
    std::vector<Algebra::Vector> prj_pt;
    prj_pt.reserve(X.size());
    h.reshape(3,3);

    for (size_t i = 0; i < X.size(); i++)
        prj_pt.emplace_back((h * X[i].hom()).hom_i());

    return vstack(prj_pt) - obs_prj_pts;
}

