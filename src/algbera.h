#pragma once

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <initializer_list>
#include <cstring>
#include <cmath>
#include <utility>

namespace Algebra
{

    class Matrix
    {
    public:
        Matrix(size_t length);
        Matrix(size_t row, size_t col);
        Matrix(const std::initializer_list<double> &lst);
        Matrix(size_t row, size_t col, const std::initializer_list<double> &lst);
        Matrix(const Matrix &mat);
        Matrix(const Matrix &mat, bool transpose);
        Matrix(Matrix &&mat);
        ~Matrix();

        Matrix &operator=(const Matrix &m);

        Matrix operator+(const Matrix &m) const;
        Matrix operator+(double s) const;
        Matrix &operator+=(const Matrix &m);
        Matrix &operator+=(double s);

        Matrix operator-(const Matrix &m) const;
        Matrix operator-(double s) const;
        Matrix &operator-=(const Matrix &m);
        Matrix &operator-=(double s);

        Matrix operator*(const Matrix &m) const;
        Matrix operator*(double s) const;
        Matrix &operator*=(const Matrix &m);
        Matrix &operator*=(double s);

        /* Matrix operator/(const Matrix& m); */
        Matrix operator/(double s) const;
        /* Matrix operator/=(const Matrix& m); */
        Matrix &operator/=(double s);

        double norm();
        Matrix hom();
        Matrix hom_i();

        double &operator[](size_t i);
        double &val(size_t row, size_t col);
        Matrix& vstack(const Matrix& mat);
        Matrix& hstack(const Matrix& mat);

        void print();
        inline void transpose() { transposed = !transposed; std::swap(row, col); }
        inline Matrix T();

    protected:
        double *m;
        size_t row, col, size;
        bool transposed;

        static const size_t buff_size = 16;
        double buff[buff_size];
    };

    inline Matrix::Matrix(size_t length)
        : row(length), col(1), size(length), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = new double[size];
    }

    inline Matrix::Matrix(size_t row, size_t col)
        : row(row), col(col), size(row * col), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = new double[size];
    }

    inline Matrix::Matrix(const std::initializer_list<double> &lst)
        : row(lst.size()), col(1), size(lst.size()), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = new double[size];
        std::copy(lst.begin(), lst.end(), m);
    }

    inline Matrix::Matrix(size_t row, size_t col, const std::initializer_list<double> &lst)
        : row(row), col(col), size(row * col), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = new double[size];
        if (lst.size() > size)
            throw std::out_of_range("Matrix : too many values in the initializer");
        else
            std::copy(lst.begin(), lst.end(), m);
    }

    inline Matrix::Matrix(const Matrix &mat)
        : row(mat.row), col(mat.col), size(mat.size), transposed(mat.transposed)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = new double[size];
        memcpy(m, mat.m, size * sizeof(double));
        // std::cout << "Matrix copy constructor\n";
    }

    inline Matrix::Matrix(const Matrix &mat, bool)
        : row(mat.row), col(mat.col), size(mat.size), transposed(mat.transposed)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = new double[size];
        memcpy(m, mat.m, size * sizeof(double));
        transpose();
    }

    inline Matrix::Matrix(Matrix &&mat)
        : row(mat.row), col(mat.col), size(mat.size), transposed(mat.transposed)
    {
        if (mat.m != mat.buff)
        {
            m = mat.m;
            mat.m = nullptr;
            mat.row = mat.col = mat.size = 0;
        }
        else
        {
            m = buff;
            memcpy(m, mat.m, size * sizeof(double));
        }
        // std::cout << "Matrix move constructor\n";
    }

    inline Matrix::~Matrix()
    {
        if (m != buff && m)
            delete[] m;
    }

    inline Matrix &Matrix::operator=(const Matrix &mat)
    {
        if (mat.size > size)
        {
            if (m != buff)
                delete[] m;
            if (mat.size > buff_size)
                m = new double[mat.size];
            else
                m = buff;
        }
        row = mat.row, col = mat.col, size = mat.size, transposed = mat.transposed;
        memcpy(m, mat.m, size * sizeof(double));
    }

    /* --------- SUM --------- */

    inline Matrix Matrix::operator+(const Matrix &m) const
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        Matrix sum(row, col);
        if (transposed == m.transposed)
        {
            for (int i = 0; i < size; i++)
                sum.m[i] = this->m[i] + m.m[i];
            sum.transposed = transposed;
        }
        else if (transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    sum.m[i * col + j] = this->m[j * row + i] + m.m[i * col + j];
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    sum.m[i * col + j] = this->m[i * col + j] + m.m[j * row + i];
        return sum;
    }

    inline Matrix Matrix::operator+(double s) const
    {
        Matrix sum(row, col);
        for (int i = 0; i < size; i++)
            sum.m[i] = m[i] + s;
        sum.transposed = transposed;
        return sum;
    }

    inline Matrix &Matrix::operator+=(const Matrix &m)
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        if (transposed == m.transposed)
            for (int i = 0; i < size; i++)
                this->m[i] += m.m[i];
        else if (transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    this->m[j * row + i] += m.m[i * col + j];
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    this->m[i * col + j] += m.m[j * row + i];
        return *this;
    }

    inline Matrix &Matrix::operator+=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] += s;
        return *this;
    }

    /* --------- SUB --------- */

    inline Matrix Matrix::operator-(const Matrix &m) const
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sub operation");
        Matrix sub(row, col);
        if (transposed == m.transposed)
        {
            for (int i = 0; i < size; i++)
                sub.m[i] = this->m[i] - m.m[i];
            sub.transposed = transposed;
        }
        else if (transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    sub.m[i * col + j] = this->m[j * row + i] - m.m[i * col + j];
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    sub.m[i * col + j] = this->m[i * col + j] - m.m[j * row + i];
        return sub;
    }

    inline Matrix Matrix::operator-(double s) const
    {
        Matrix sub(row, col);
        for (int i = 0; i < size; i++)
            sub.m[i] = m[i] - s;
        sub.transposed = transposed;
        return sub;
    }

    inline Matrix &Matrix::operator-=(const Matrix &m)
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in subtraction operation");
        if (transposed == m.transposed)
            for (int i = 0; i < size; i++)
                this->m[i] -= m.m[i];
        else if (transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    this->m[j * row + i] -= m.m[i * col + j];
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    this->m[i * col + j] -= m.m[j * row + i];
        return *this;
    }

    inline Matrix &Matrix::operator-=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] -= s;
        return *this;
    }

    /* --------- MULT --------- */

    inline Matrix Matrix::operator*(const Matrix &m) const
    {
        if (col != m.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");
        Matrix mul(row, m.col);
        memset(mul.m, 0, row * m.col * sizeof(double));
        if (!transposed && !m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        mul.m[i * mul.col + j] += this->m[i * col + k] * m.m[k * m.col + j];
        else if (!transposed && m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        mul.m[i * mul.col + j] += this->m[i * col + k] * m.m[k + j * m.row];
        else if (transposed && !m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        mul.m[i * mul.col + j] += this->m[i + k * row] * m.m[k * m.col + j];
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        mul.m[i * mul.col + j] += this->m[i + k * row] * m.m[k + j * m.row];
        return mul;
    }

    inline Matrix Matrix::operator*(double s) const
    {
        Matrix mul(row, col);
        for (int i = 0; i < size; i++)
            mul.m[i] = m[i] * s;
        mul.transposed = transposed;
        return mul;
    }

    inline Matrix &Matrix::operator*=(const Matrix &m)
    {
        if (col != m.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");

        double *new_m;
        if (this->m != buff && col * m.row <= buff_size)
            new_m = buff;
        else
            new_m = new double[row * m.col];

        memset(new_m, 0, row * m.col * sizeof(double));

        if (!transposed && !m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        new_m[i * m.col + j] += this->m[i * col + k] * m.m[k * m.col + j];
        else if (!transposed && m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        new_m[i * m.col + j] += this->m[i * col + k] * m.m[k + j * m.row];
        else if (transposed && !m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        new_m[i * m.col + j] += this->m[i + k * row] * m.m[k * m.col + j];
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                    for (int k = 0; k < col; k++)
                        new_m[i * m.col + j] += this->m[i + k * row] * m.m[k + j * m.row];

        if (this->m != buff)
            delete[] this->m;
        this->m = new_m;
        return *this;
    }

    inline Matrix &Matrix::operator*=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] *= s;
        return *this;
    }

    /* --------- MULT --------- */

    inline Matrix Matrix::operator/(double s) const
    {
        Matrix div(row, col);
        for (int i = 0; i < size; i++)
            div.m[i] = m[i] / s;
        div.transposed = transposed;
        return div;
    }

    inline Matrix &Matrix::operator/=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] /= s;
        return *this;
    }

    double Matrix::norm()
    {
        double sum = 0.;
        for (int i = 0; i < size; i++)
            sum += m[i] * m[i];
        return sqrt(sum);
    }

    inline Matrix Matrix::hom()
    {
        Matrix hv(size + 1);
        memcpy(hv.m, m, size * sizeof(double));
        hv.m[size] = 1;
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    inline Matrix Matrix::hom_i()
    {
        Matrix hv(size - 1);
        for (size_t i = 0; i < size - 1; i++)
            hv.m[i] = m[i] / m[size - 1];
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    inline double &Matrix::operator[](size_t i)
    {
        return m[i];
    }

    inline double &Matrix::val(size_t row, size_t col)
    {
        if (row >= this->row || col >= this->col)
            throw std::out_of_range("Matrix[] : index is out of range");
        if (transposed)
            return m[row + col * this->row];
        else
            return m[row * this->col + col];
    }

    inline Matrix &Matrix::vstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix columns not matching for vertical stacking");

        double* new_buf;

        if (col * (row + mat.row) < buff_size && m != buff)
            new_buf = buff;
        else
            new_buf = new double[col * (row + mat.row)];
        
        int row_off = transposed ? 1 : col, col_off = transposed ? row : 1;
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                new_buf[i*col + j] = m[i*row_off + j*col_off];

        row_off = transposed ? 1 : mat.col, col_off = transposed ? mat.row : 1;
        for (int i = 0; i < mat.row; i++)
            for (int j = 0; j < col; j++)
                new_buf[size + i*col + j] = mat.m[i*row_off + j*col_off];

        if (m != buff) delete[] m;
        m = new_buf;
        row += mat.row;
        size += mat.size;
        transposed = false;

        return *this;
    }

    inline Matrix &Matrix::hstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix rows not matching for horizontal stacking");

        double* new_buf;

        if (col * (row + mat.row) < buff_size && m != buff)
            new_buf = buff;
        else
            new_buf = new double[row * (col + mat.col)];
        
        int row_off = transposed ? 1 : col, col_off = transposed ? row : 1;
        for (int j = 0; j < col; j++)
            for (int i = 0; i < row; i++)
                new_buf[i + j*row] = m[i*row_off + j*col_off];

        row_off = transposed ? 1 : mat.col, col_off = transposed ? mat.row : 1;
        for (int j = 0; j < col; j++)
            for (int i = 0; i < mat.row; i++)
                new_buf[size + i + j*row] = mat.m[i*row_off + j*col_off];

        if (m != buff) delete[] m;
        m = new_buf;
        col += mat.col;
        size += mat.size;
        transposed = true;
        
        return *this;
    }

    inline void Matrix::print()
    {
        std::cout << "[";
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < this->col; j++)
                if (transposed)
                    std::cout << m[j * this->row + i] << " ";
                else
                    std::cout << m[i * this->col + j] << " ";
            if (i < this->row - 1) std::cout << "\n ";
        }
        std::cout << "]\n";
    }

    inline Matrix Matrix::T()
    {
        return {*this, true};
    }

    using Vector = Matrix;

#ifdef ALGEBRA_SHORT_NAMES
    using Vec = Matrix;
    using Mat = Matrix;
#endif

}