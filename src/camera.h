#include <iostream>
#include "algbera.h"
#include "calibration.h"

namespace MoCap
{
    class Camera
    {
    public:
        Camera(int width, int height, double pixel_size);
        ~Camera();

        double calibrateIntrinsics(int layout_x, int layout_y, double quad_size, const std::vector<std::vector<Algebra::Vector>>& view_point_sets);
        double calibrateExtrinsics();

        void print();

    private:
        Algebra::Matrix A;
        Algebra::Matrix Rt;
        Algebra::Vector k, p;
        double pix_size;
        double focal_lenght;
        int width, height;

    };
    
    Camera::Camera(int width, int height, double pixel_size)
        : A(Algebra::identity(3)), Rt(Algebra::identity(3,4)), k(3),
          p(2), pix_size(pixel_size), focal_lenght(0.), width(width), height(height)
    {
    }
    
    Camera::~Camera()
    {
    }

    inline double Camera::calibrateIntrinsics(int layout_x, int layout_y, double quad_size, const std::vector<std::vector<Algebra::Vector>>& view_point_sets)
    {
        using namespace Algebra;
        using namespace std::placeholders;
        std::vector<Vector> world_points = generateCheckerboardCoords(layout_x, layout_y, quad_size);
        auto [Nx, Nx_i] = normalize2Dset(world_points);

        /* Homographies estimation */

        std::vector<Matrix> H;
        H.reserve(world_points.size());

        for (auto imgpt : view_point_sets)
        {
            auto [Nu, Nu_i] = normalize2Dset(imgpt);

            int n_pt = layout_x * layout_y;
            Matrix M(2 * n_pt, 9);

            for (int i = 0; i < imgpt.size(); i++)
            {
                double Xi = world_points[i](0);
                double Yi = world_points[i](1);
                double ui = imgpt[i](0);
                double vi = imgpt[i](1);

                M.setRow(i * 2,     { -Xi, -Yi, -1, 0, 0, 0, ui * Xi, ui * Yi, ui});
                M.setRow(i * 2 + 1, { 0, 0, 0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi});
            }

            auto[U, S, Vt] = M.svd();
            Vector h = Vt.getRow(-1).reshape(9,1);

            Vector data = vstack(imgpt);
            h = minimizelm(std::bind(multiPointProjectionError, world_points, _1, data), h, 
                           std::bind(multiPointProjectionErrorJacobian, world_points, _1)); 

            h.reshape(3, 3);
            H.push_back(Nu_i * h * Nx);

            unnormalizeSet(imgpt, Nu_i);
        }

        unnormalizeSet(world_points, Nx_i);

        /* Intrinsics estimation */

        Vector V(0,6);
        for (auto& h : H)
            V.vstack({vpq(0,1, h), vpq(0,0, h) - vpq(1,1, h)});

        auto[U, S, Vt] = V.svd();
        Vector b = Vt.getRow(-1);
        b.transpose();
        Matrix B = {3,3, {
            b(0),b(1),b(3), b(1),b(2),b(4), b(3),b(4),b(5)
        }};
        
        /* Find A from H, and intrinsic parameters */

        if (b(0) < 0 || b(2) < 0 || b(5) < 0)
            B *= -1.;
        
        Matrix A_i = B.cholesky().T();
        A_i /= A_i[2][2];
        Matrix A_ = A_i;
        upperTriangInvert(A_);

        /* Alternative way for extracting parameters */

        double w = b(0)*b(2)*b(5) - pow(b(1), 2)*b(5) - b(0)*pow(b(4), 2) + 2*b(1)*b(3)*b(4) - b(2)*pow(b(3), 2);
        double d = b(0)*b(2) - pow(b(1), 2);

        double alpha = sqrt(w / (d * b(0)));
        double beta  = sqrt(w / (d*d) * b(0));
        double gamma = sqrt(w / (d*d  * b(0))) * b(1);
        double u_c   = (b(1)*b(4) - b(2)*b(3)) / d;
        double v_c   = (b(1)*b(3) - b(0)*b(4)) / d;
        Vector prjcen({u_c, v_c});

        /* Extrinsics estimation */

        std::vector<Matrix> Rt;
        Rt.reserve(H.size());
        for (auto& h : H)
        {
            double lam = 1. / (A_i * h.getCol(0)).norm();
            Matrix r0r1t = (lam * A_i * h);
            Vector r2 = cross(r0r1t.getCol(0), r0r1t.getCol(1));
            Rt.emplace_back(hstack({r0r1t.subMatrix(0,0,-1,1), r2, r0r1t.getCol(2)}));
        }

        /* Distortion parameters guess */

        Matrix D_mat(0,3), d_dot(0,1);
        for (size_t i = 0; i < view_point_sets.size(); i++)
            for (size_t j = 0; j < view_point_sets[i].size(); j++)
            {
                Vector prjpt = (H[i] * world_points[j].hom()).hom_i();
                double r = prjpt.norm();
                Vector rv = {1,3, {pow(r,2), pow(r,4), pow(r,6)}};
                d_dot.vstack(view_point_sets[i][j] - prjpt);
                D_mat.vstack((view_point_sets[i][j] - prjpt) * rv);
            }
        
        Vector k_ = D_mat.solve(d_dot);
        Vector p_({0,0});

        /* Minimize reprojection error */

        Vector a({alpha,beta,gamma,u_c,v_c,k_(0),k_(1),k_(2),p_(0),p_(1)});
        std::vector<Vector> W;
        W.reserve(Rt.size());
        for (auto& rt : Rt)
            W.emplace_back(matrixToRodrigues(rt.subMatrix(0,0,2,2)).vstack(rt.getCol(3)));

        Vector P = a;
        P.vstack(W);
        Vector obs_obj_pt(view_point_sets.size()*view_point_sets[0].size()*2);
        for (size_t i = 0; i < view_point_sets.size(); i++)
            obs_obj_pt.setSubMatrix(vstack(view_point_sets[i]), i*view_point_sets[0].size()*2);

        Vector P_opt = minimizelm(std::bind(multiPointProjectionWDistError, world_points, obs_obj_pt, _1), P, 
                                  std::bind(multiPointProjectionWDistErrorJacobian, world_points, _1));

        A[0][0] = P_opt(0), A[0][1] = P_opt(2), A[0][2] = P_opt(3);
        A[1][0] = 0       , A[1][1] = P_opt(1), A[1][2] = P_opt(4);
        A[2][0] = 0       , A[2][1] = 0       , A[2][2] = 1       ;

        k(0) = P_opt(5), k(1) = P_opt(6), k(2) = P_opt(7);
        p(0) = P_opt(8), p(1) = P_opt(9);

        focal_lenght = P_opt(0) * pix_size;

        return multiPointProjectionWDistError(world_points, obs_obj_pt, P_opt).pnorm(1)/obs_obj_pt.getSize();
    }

    inline double Camera::calibrateExtrinsics()
    {
        return 0.0;
    }

    inline void Camera::print()
    {
        std::cout << "\nCamera sensor\n";
        std::cout << "width: " << width << ", height: " << height << "\n";
        std::cout << "focal length: " << focal_lenght * 1e3 << "mm\n";
        std::cout << "pixel size: " << pix_size * 1e6 << "um\n";

        std::cout << "\nParameters\n";
        std::cout << "A = \n";
        A.print();
        std::cout << "Rt = \n";
        Rt.print();
        std::cout << "k = \n";
        k.print();
        std::cout << "p = \n";
        p.print();
        std::cout << "\n";
    }

} // namespace MoCap
