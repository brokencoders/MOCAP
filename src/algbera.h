#pragma once

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <initializer_list>
		


namespace Algebra {

    class Matrix {
    public: 
        Matrix(size_t row, size_t col, std::initializer_list<double> lst = {});
        ~Matrix();

        Matrix operator+(const Matrix& m);
        Matrix operator+(double s);
        Matrix operator+=(const Matrix& m);
        Matrix operator+=(double s);

        Matrix operator-(const Matrix& m);
        Matrix operator-(double s);
        Matrix operator-=(const Matrix& m);
        Matrix operator-=(double s);

        Matrix operator*(const Matrix& m);
        Matrix operator*(double s);
        Matrix operator*=(const Matrix& m);
        Matrix operator*=(double s);


        Matrix operator/(const Matrix& m);
        Matrix operator/(double s);
        Matrix operator/=(const Matrix& m);
        Matrix operator/=(double s);

        double& val(size_t row, size_t col);

        void print();
        inline void transpose() { transposed = !transposed; }
    private:
        double* m;
        int row, col;
        bool transposed;
    };

    Matrix::Matrix(size_t row, size_t col, std::initializer_list<double> lst) 
        :transposed(false), row(row), col(col)
    {
        m = new double[row * col];
        if (lst.size() > row * col)
            throw std::out_of_range("Matrix : too many values in the initializer");
        else std::copy(lst.begin(), lst.end(), m);
    }

    Matrix::~Matrix()
    {
        if (m) delete m; 
    }

    inline Matrix Matrix::operator+(const Matrix &m)
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        Matrix sum(row, col);
        for (int i = row * col; i >= 0; i--)
            sum.m[i] = this->m[i] + m.m[i];
        return sum;
    }

    inline Matrix Matrix::operator+(double s)
    {
        return Matrix(0, 0);
    }

    inline Matrix Matrix::operator+=(const Matrix &m)
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        for (int i = row * col; i >= 0; i--)
            this->m[i] += m.m[i];
        return *this;
    }

    double& Matrix::val(size_t row, size_t col)
    {
        if (row >= this->row || col >= this->col) 
            throw std::out_of_range("Matrix[] : index is out of range");
        return m[row * this->col + col];
    }

    void Matrix::print() 
    {
        std::cout << "[";
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < this->col; j++)
                if (transposed)
                    std::cout << m[j*this->row + i] << " ";
                else
                    std::cout << m[i*this->col + j] << " ";
            std::cout << "\n ";
        }
        std::cout << "]";
    }

    #ifndef ALGEBRA_DEFAULT_NAMES

    using Vec2 = Vec<2>;
    using Vec3 = Vec<3>;
    using Vec4 = Vec<4>;

    using Mat2x2 = Matrix<2, 2>; 
    using Mat3x3 = Matrix<3, 3>; 
    using Mat3x4 = Matrix<3, 4>; 
    using Mat4x4 = Matrix<4, 4>; 

    #endif
}