#pragma once 

#include <tuple>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>

#include "algbera.h"


inline std::vector<Algebra::Vector> generateCheckerboardCoords(int layout_x, int layout_y, double size)
{
    std::vector<Algebra::Vector> checkerboard(layout_x * layout_y, Algebra::Vector(2));
    int x = -1, y = -1;

    for (auto& pt : checkerboard)
    {
        x = (x + 1) % layout_x;
        if (x == 0) y++;
        pt[0][0] = x * size;
        pt[0][1] = y * size;
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
             s, 0, -s * x_bar(0),
             0, s, -s * x_bar(1),
             0, 0, 1
            }},
            { 3, 3, {
             1/s,   0, x_bar(0),
             0,   1/s, x_bar(1),
             0,     0,        1
            }}};

}

inline void unnormalizeSet(std::vector<Algebra::Vector>& X, const Algebra::Matrix& N_inverse)
{
    for(auto& x : X)
        x = (N_inverse * x.hom()).hom_i();
}

inline
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
        for (size_t i = 0; i < A.rows(); i++) A[i][i] += u;
        Algebra::Vector h_lm = A.solve(-g);
        if (h_lm.norm() <= e2 * (x.norm() + e2))
            found = true;
        else
        {
            Algebra::Vector x_new = x + h_lm;
            Algebra::Vector fx = f(x);
            Algebra::Vector fx_new = f(x_new);
            double o = (double)(fx.T() * fx - fx_new.T() * fx_new) / (h_lm.T() * (u * h_lm - g));
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
                for (size_t i = 0; i < A.rows(); i++) A[i][i] -= u;
                u = u * v;
                v *= 2.0;
            }
        }
    }
    return x;
}

inline
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

inline
Algebra::Matrix multiPointProjectionErrorJacobian(const std::vector<Algebra::Vector>& X,
                                                  Algebra::Vector h)
{
    size_t N = X.size();
    Algebra::Matrix J(2*N, 9);
    h.reshape(3,3);
    for (size_t i = 0; i < N; i++)
    {
        const double x = X[i].x, y = X[i].y;
        Algebra::Vector s = h*X[i].hom();
        const double w = s.z;
        J.setRow(2*i,   {x/w, y/w, 1/w, 0, 0, 0, -s.x*x/(w*w), -s.x*y/(w*w), -s.x/(w*w)});
        J.setRow(2*i+1, {0, 0, 0, x/w, y/w, 1/w, -s.y*x/(w*w), -s.y*y/(w*w), -s.y/(w*w)});
    }
    return J;
}

inline Algebra::Vector vpq(size_t p, size_t q, const Algebra::Matrix& H)
{
    return {1, 6, {
        H[0][p]*H[0][q],
        H[0][p]*H[1][q] + H[1][p]*H[0][q],
        H[1][p]*H[1][q],
        H[2][p]*H[0][q] + H[0][p]*H[2][q],
        H[2][p]*H[1][q] + H[1][p]*H[2][q],
        H[2][p]*H[2][q],
    }};
}

inline Algebra::Vector project(const Algebra::Matrix& A,    // intrinsics matrix
                               const Algebra::Matrix& Rt,   // rotation matrix
                               const Algebra::Matrix& k,    // radial distortion coefficents
                               const Algebra::Matrix& p,    // tangential distortion coefficents
                               const Algebra::Vector& X)    // 3d point
{
    using namespace Algebra;
    Vector pt = Rt * X.hom();
    pt /= pt(2);
    double r2 = pt.x*pt.x + pt.y*pt.y;
    double r4 = r2*r2;
    double r6 = r4*r2;
    double radial = k(0)*r2 + k(1)*r4 + k(2)*r6;
    pt.x += pt.x * radial + p(1)*(r2 + 2*pow(pt.x, 2)) + 2*p(0)*pt.x*pt.y;
    pt.y += pt.y * radial + p(0)*(r2 + 2*pow(pt.y, 2)) + 2*p(1)*pt.x*pt.y;
    return (A * pt).hom_i();
}

inline
Algebra::Vector multiPointProjectionWDistError(const std::vector<Algebra::Vector>& X,   // 3d points to be projected
                                               const Algebra::Vector& obs_prj_pts,      // observed projected points
                                               Algebra::Vector P)                       // stacked projection parameters
{
    using namespace Algebra;
    Matrix A(3,3, {P(0),P(2),P(3), 0,P(1),P(4), 0,0,1});
    Matrix Rt(3,4);
    Vector k = P.subMatrix(5,0,7);
    Vector p = P.subMatrix(8,0,9);
    size_t N = (P.getSize()-10) / 6;
    Vector v(N*X.size()*2);

    for (size_t i = 0; i < N; i++)
    {
        Rt.setSubMatrix(rodriguesToMatrix(P.subMatrix(10+i*6, 0, 12+i*6)));
        Rt.setSubMatrix(P.subMatrix(13+i*6, 0, 15+i*6), 0,3);

        for (size_t j = 0; j < X.size(); j++)
        {
            Vector pt({X[j](0), X[j](1), 0});
            v.setSubMatrix(
                project(A, Rt, k, p, pt), i*X.size()*2 + j*2);
        }
        
    }
    
    v -= obs_prj_pts;
    return v;
}

inline
Algebra::Matrix multiPointProjectionWDistErrorJacobian(const std::vector<Algebra::Vector>& X,
                                                       Algebra::Vector P)
{
    using namespace Algebra;
    Matrix A(3,3, {P(0),P(2),P(3), 0,P(1),P(4), 0,0,1});
    Vector k({P(5),P(6),P(7)});
    Vector p({P(8),P(9)});
    Matrix Rt(3,4), dRt(3,4);
    size_t N = (P.getSize()-10) / 6;
    Matrix jac(N*X.size()*2, P.getSize());
    Matrix tinyA(2,2, {P(0), P(2), 0, P(1)});   // [[alpha,gamma], [0,beta]]
    
    for (size_t i = 10, n = 0; i < P.getSize(); i+=6, n++)
    {
        Vector w = P.subMatrix(i,0,i+5);
        Rt.setSubMatrix(rodriguesToMatrix(P.subMatrix(i, 0, i+2)));
        Rt.setSubMatrix(P.subMatrix(i+3, 0, i+5), 0,3);
        for (size_t j = 0; j < X.size(); j++)
        {
            Vector x({X[j](0), X[j](1), 0});
            Vector pt = (Rt * x.hom()).hom_i();
            double r2 = pt.x*pt.x + pt.y*pt.y;
            double r4 = r2*r2;
            double r6 = r4*r2;
            Vector r(1,3, {r2,r4,r6});
            double radial = k(0)*r2 + k(1)*r4 + k(2)*r6;
            double xd = pt.x + pt.x * radial + p(1)*(r2 + 2*pow(pt.x, 2)) + 2*p(0)*pt.x*pt.y;
            double yd = pt.y + pt.y * radial + p(0)*(r2 + 2*pow(pt.y, 2)) + 2*p(1)*pt.x*pt.y;

            // Intrinsics derivatives
            // [ xd,  0, yd, 1, 0, tinyA * pt * [r2,r4,r6],     2xy, r2+2x^2 ]
            // [  0, yd,  0, 0, 1,   ------- 2x3 -------  , r2+2y^2,     2xy ]
            auto jr0 = jac[n*X.size()*2 + j*2], jr1 = jac[n*X.size()*2 + j*2 + 1];
            jr0[1] = jr0[4] = jr1[0] = jr1[2] = jr1[3] = 0.;
            jr0[3] = jr1[4] = 1.;
            jr0[0] = xd, jr0[2] = jr1[1] = yd;
            jr0[8] = jr1[9] = 2*pt.x*pt.y;
            jr0[9] = r2 + 2*pt.x*pt.x;
            jr1[8] = r2 + 2*pt.y*pt.y;
            jac.setSubMatrix(tinyA * pt * r, n*X.size()*2 + 2*j, 5);

            // Rt derivatives
            for (size_t l = 0; l < 6; l++)
            {
                double delta = 1.5e-8 * std::max(std::abs(w(i)), 1.);
                double wl = w(l);
                w(l) += delta;
                if (l < 4)
                    dRt.setSubMatrix(rodriguesToMatrix(w.subMatrix(0,0, 2)));
                if (l == 0 || l > 2)
                    dRt.setSubMatrix(P.subMatrix(i+3, 0, i+5), 0,3);
                w(l) = wl;
                jac.setSubMatrix((project(A, dRt, k, p, x) - project(A, Rt, k, p, x))/delta,  n*X.size()*2 + 2*j, i+l);
            }
            

            // Fill in unused views

            std::fill(jr0 + 10, jr0 + i, 0.);
            std::fill(jr1 + 10, jr1 + i, 0.);
            std::fill(jr0 + i + 6, jr0 + P.getSize(), 0.);
            std::fill(jr1 + i + 6, jr1 + P.getSize(), 0.);
        }
        
    }
    //jac.print();
    return jac;
}

