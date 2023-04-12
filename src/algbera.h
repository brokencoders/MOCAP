#pragma once

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <initializer_list>
#include <cstring>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <immintrin.h>
#include <tuple>

namespace Algebra
{
    inline size_t closest_mult(size_t n, size_t mult)
    {
        if (n % mult != 0)
            return n + mult - n % mult;
        else 
            return n;
    }

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
        Matrix operator-() const;
        Matrix operator-(double s) const;
        Matrix &operator-=(const Matrix &m);
        Matrix &operator-=(double s);

        Matrix operator*(const Matrix &m) const;
        Matrix operator*(double s) const;
        Matrix &operator*=(const Matrix &m);
        Matrix &operator*=(double s);

        Matrix operator/(const Matrix& m);
        Matrix operator/(double s) const;
        /* Matrix operator/=(const Matrix& m); */
        Matrix &operator/=(double s);

        operator double();

        double norm();

        double max();
        double absMax();

        Matrix hom() const;
        Matrix hom_i() const;

        Matrix solve(const Matrix& b);
        std::tuple<Matrix, Matrix, Matrix> svd();

        void reshape(size_t row, size_t col);

        double &operator[](size_t i);
        double &val(size_t row, size_t col);
        Matrix& vstack(const Matrix& mat);
        Matrix& hstack(const Matrix& mat);

        void setCol(int64_t c, const std::initializer_list<double> &lst);
        void setRow(int64_t r, const std::initializer_list<double> &lst);
        Matrix getCol(int64_t c);
        Matrix getRow(int64_t r);

        void print();
        void transpose() { transposed = !transposed; std::swap(row, col); }
        Matrix T();

    private:
        double *m;
        size_t row, col, size;
        bool transposed;

        static const size_t buff_size = 16;
        alignas(32) double buff[buff_size];

    public:
        const size_t& rows = row;
        const size_t& cols = col;

        friend void zero(Matrix&);
        friend Matrix identity(size_t);
        friend Matrix operator+(double, const Matrix&);
        friend Matrix operator-(double, const Matrix&);
        friend Matrix operator*(double, const Matrix&);
        friend Matrix vstack(const std::vector<Matrix>& mats);
        friend Matrix hstack(const std::vector<Matrix>& mats);
    };

    using Vector = Matrix;

#ifdef ALGEBRA_IMPL

    Matrix::Matrix(size_t length)
        : row(length), col(1), size(length), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
    }

    Matrix::Matrix(size_t row, size_t col)
        : row(row), col(col), size(row * col), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
    }

    Matrix::Matrix(const std::initializer_list<double> &lst)
        : row(lst.size()), col(1), size(lst.size()), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        std::copy(lst.begin(), lst.end(), m);
    }

    Matrix::Matrix(size_t row, size_t col, const std::initializer_list<double> &lst)
        : row(row), col(col), size(row * col), transposed(false)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        if (lst.size() > size)
            throw std::out_of_range("Matrix : too many values in the initializer");
        else
            std::copy(lst.begin(), lst.end(), m);
    }

    Matrix::Matrix(const Matrix &mat)
        : row(mat.row), col(mat.col), size(mat.size), transposed(mat.transposed)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        memcpy(m, mat.m, size * sizeof(double));
        // std::cout << "Matrix copy constructor\n";
    }

    Matrix::Matrix(const Matrix &mat, bool)
        : row(mat.row), col(mat.col), size(mat.size), transposed(mat.transposed)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        memcpy(m, mat.m, size * sizeof(double));
        transpose();
    }

    Matrix::Matrix(Matrix &&mat)
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

    Matrix::~Matrix()
    {
        if (m != buff && m)
            std::free(m);
    }

    Matrix &Matrix::operator=(const Matrix &mat)
    {
        if (mat.size > size)
        {
            if (m != buff)
                std::free(m);
            if (mat.size > buff_size)
                m = (double*)std::aligned_alloc(32, mat.size * sizeof(double));
            else
                m = buff;
        }
        row = mat.row, col = mat.col, size = mat.size, transposed = mat.transposed;
        memcpy(m, mat.m, size * sizeof(double));
        return *this;
    }

    /* --------- SUM --------- */

    Matrix Matrix::operator+(const Matrix &m) const
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

    Matrix Matrix::operator+(double s) const
    {
        Matrix sum(row, col);
        for (int i = 0; i < size; i++)
            sum.m[i] = m[i] + s;
        sum.transposed = transposed;
        return sum;
    }

    Matrix operator+(double s, const Matrix& mat)
    {
        Matrix sum(mat.rows, mat.cols);
        for (int i = 0; i < mat.size; i++)
            sum.m[i] = mat.m[i] + s;
        sum.transposed = mat.transposed;
        return sum;
    }

    Matrix &Matrix::operator+=(const Matrix &m)
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

    Matrix &Matrix::operator+=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] += s;
        return *this;
    }

    /* --------- SUB --------- */

    Matrix Matrix::operator-(const Matrix &m) const
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

    inline Matrix Matrix::operator-() const
    {
        Matrix sub(row, col);
        for (size_t i = 0; i < size; i++)
            sub.m[i] = -m[i];
        return sub;        
    }

    Matrix operator-(double s, const Matrix& mat)
    {
        Matrix sub(mat.rows, mat.cols);
        for (int i = 0; i < mat.size; i++)
            sub.m[i] = mat.m[i] - s;
        sub.transposed = mat.transposed;
        return sub;
    }

    Matrix Matrix::operator-(double s) const
    {
        Matrix sub(row, col);
        for (int i = 0; i < size; i++)
            sub.m[i] = m[i] - s;
        sub.transposed = transposed;
        return sub;
    }

    Matrix &Matrix::operator-=(const Matrix &m)
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

    Matrix &Matrix::operator-=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] -= s;
        return *this;
    }

    /* --------- MULT --------- */

    Matrix Matrix::operator*(const Matrix &m) const
    {
        if (col != m.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");
        Matrix mul(row, m.col);
        if (!transposed && !m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                {
                    double& sum = mul.m[i * mul.col + j];
                    sum = 0;
                    for (int k = 0; k < col; k++)
                        sum += this->m[i * col + k] * m.m[k * m.col + j];
                }
        else if (!transposed && m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                {
                    double& sum = mul.m[i * mul.col + j];
                    sum = 0;
                    for (int k = 0; k < col; k++)
                        sum += this->m[i * col + k] * m.m[k + j * m.row];
                }
        else if (transposed && !m.transposed)
        {
            for (int j = 0; j < m.col; j++)
                for (int i = 0; i < row; i++)
                    mul.m[i + j * mul.row] = this->m[i] * m.m[j];

            for (int k = 1; k < col; k++)
                for (int j = 0; j < m.col; j++)
                    for (int i = 0; i < row; i++)
                        mul.m[i + j * mul.row] += this->m[i + k * row] * m.m[k * m.col + j];
            mul.transposed = true;
        }
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                {
                    double& sum = mul.m[i * mul.col + j];
                    sum = 0;
                    for (int k = 0; k < col; k++)
                        sum += this->m[i + k * row] * m.m[k + j * m.row];
                }
        return mul;
    }

    Matrix Matrix::operator*(double s) const
    {
        Matrix mul(row, col);
        for (int i = 0; i < size; i++)
            mul.m[i] = m[i] * s;
        mul.transposed = transposed;
        return mul;
    }

    Matrix operator*(double s, const Matrix& mat)
    {
        Matrix mul(mat.rows, mat.cols);
        for (int i = 0; i < mat.size; i++)
            mul.m[i] = mat.m[i] * s;
        mul.transposed = mat.transposed;
        return mul;
    }

    Matrix &Matrix::operator*=(const Matrix &m)
    {
        if (col != m.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");

        double *new_m;
        if (this->m != buff && col * m.row <= buff_size)
            new_m = buff;
        else
            new_m = (double*)std::aligned_alloc(32, row * m.col * sizeof(double));

        if (!transposed && !m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < col; k++)
                        sum += this->m[i * col + k] * m.m[k * m.col + j];
                    new_m[i * m.col + j] = sum;
                }
        else if (!transposed && m.transposed)
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < col; k++)
                        sum += this->m[i * col + k] * m.m[k + j * m.row];
                    new_m[i * m.col + j] = sum;
                }
        else if (transposed && !m.transposed)
        {
            for (int j = 0; j < m.col; j++)
                for (int i = 0; i < row; i++)
                    new_m[i + j * row] = this->m[i] * m.m[j];

            for (int k = 1; k < col; k++)
                for (int j = 0; j < m.col; j++)
                    for (int i = 0; i < row; i++)
                        new_m[i + j * row] += this->m[i + k * row] * m.m[k * m.col + j];
            transposed = true;
        }
        else
            for (int i = 0; i < row; i++)
                for (int j = 0; j < m.col; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < col; k++)
                        sum += this->m[i + k * row] * m.m[k + j * m.row];
                    new_m[i * m.col + j] = sum;
                }

        if (this->m != buff)
            free(this->m);
        this->m = new_m;
        return *this;
    }

    Matrix &Matrix::operator*=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] *= s;
        return *this;
    }

    /* --------- DIV --------- */

    inline Matrix Matrix::operator/(const Matrix &mat)
    {
        if (row != mat.row || col != mat.col)
            throw std::length_error("Matrix sizes not matching in division operation");
        Matrix div(row, col);
        if (transposed == mat.transposed)
            for (size_t i = 0; i < size; i++)
                div.m[i] = m[i] / mat.m[i];
        else if (!transposed)
            for (size_t i = 0; i < row; i++)
                for (size_t j = 0; j < col; j++)
                    div.m[i * col + j] = m[i * col + j] / mat.m[j * row + i];
        else
            for (size_t i = 0; i < row; i++)
                for (size_t j = 0; j < col; j++)
                    div.m[i * col + j] = m[i + j * row] / mat.m[i * col + j];

        div.transposed = transposed;
        return div;
    }

    Matrix Matrix::operator/(double s) const
    {
        Matrix div(row, col);
        for (int i = 0; i < size; i++)
            div.m[i] = m[i] / s;
        div.transposed = transposed;
        return div;
    }

    Matrix &Matrix::operator/=(double s)
    {
        for (int i = 0; i < size; i++)
            m[i] /= s;
        return *this;
    }

    Matrix::operator double()
    {
        if (row != 1 || col != 1)
            throw std::length_error("Matrix is not 1x1 in double cast");
        return m[0];
    }

    double Matrix::norm()
    {
        double sum = 0.;
        for (int i = 0; i < size; i++)
            sum += m[i] * m[i];
        return sqrt(sum);
    }

    double Matrix::max()
    {
        double max = m[0];
        for(int i = 0; i < size; i++)
            if(m[i] > max) max = m[i];
        return max;
    }


    double Matrix::absMax()
    {
        double max = 0;
        for(int i = 0; i < size; i++)
            if(std::abs(m[i]) > max) max = std::abs(m[i]);
        return max;
    }


    Matrix Matrix::hom() const
    {
        Matrix hv(size + 1);
        memcpy(hv.m, m, size * sizeof(double));
        hv.m[size] = 1;
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    Matrix Matrix::hom_i() const
    {
        Matrix hv(size - 1);
        for (size_t i = 0; i < size - 1; i++)
            hv.m[i] = m[i] / m[size - 1];
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    Matrix Matrix::solve(const Matrix &b)
    {
        return Matrix(0);
    }

    std::tuple<Matrix, Matrix, Matrix> Matrix::svd()
    {
        return {Matrix(0), Matrix(0), Matrix(0)};
    }

    void Matrix::reshape(size_t row, size_t col)
    {
        if (size != row * col)
            throw std::length_error("Matrix columns not matching for vertical stacking");

        if (!transposed)
            this->row = row, this->col = col;
        else
        {
            // todo
        }
    }

    double &Matrix::operator[](size_t i)
    {
        return m[i];
    }

    double &Matrix::val(size_t row, size_t col)
    {
        if (row >= this->row || col >= this->col)
            throw std::out_of_range("Matrix[] : index is out of range");
        if (transposed)
            return m[row + col * this->row];
        else
            return m[row * this->col + col];
    }

    Matrix &Matrix::vstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix columns not matching for vertical stacking");

        double* new_buf;

        if (col * (row + mat.row) < buff_size && m != buff)
            new_buf = buff;
        else
            new_buf = (double*)std::aligned_alloc(32, col * (row + mat.row) * sizeof(double));
        
        int row_off = transposed ? 1 : col, col_off = transposed ? row : 1;
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                new_buf[i*col + j] = m[i*row_off + j*col_off];

        row_off = transposed ? 1 : mat.col, col_off = transposed ? mat.row : 1;
        for (int i = 0; i < mat.row; i++)
            for (int j = 0; j < col; j++)
                new_buf[size + i*col + j] = mat.m[i*row_off + j*col_off];

        if (m != buff) std::free(m);
        m = new_buf;
        row += mat.row;
        size += mat.size;
        transposed = false;

        return *this;
    }

    Matrix vstack(const std::vector<Matrix>& mats)
    {
        if (mats.empty()) return Matrix(0, 0);
        size_t col = mats[0].col;
        size_t row = 0;
        for (auto& mat : mats)
        {
            if (mat.col != col)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            row += mat.row;
        }

        Matrix stack(row, col);
        size_t off = 0;
        for (auto& mat : mats)
        {
            if (mat.transposed)
                for (size_t i = 0; i < mat.row; i++)
                    for (size_t j = 0; j < col; j++)
                        stack.m[i * col + j + off] = mat.m[i + j * row];
            else
                for (size_t i = 0; i < mat.row; i++)
                    for (size_t j = 0; j < col; j++)
                        stack.m[i * col + j + off] = mat.m[i * col + j];
            
            off += mat.row * col;
        }
        
        return stack;
    }

    Matrix &Matrix::hstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix rows not matching for horizontal stacking");

        double* new_buf;

        if (col * (row + mat.row) < buff_size && m != buff)
            new_buf = buff;
        else
            new_buf = (double*)std::aligned_alloc(32, row * (col + mat.col) * sizeof(double));
        
        int row_off = transposed ? 1 : col, col_off = transposed ? row : 1;
        for (int j = 0; j < col; j++)
            for (int i = 0; i < row; i++)
                new_buf[i + j*row] = m[i*row_off + j*col_off];

        row_off = transposed ? 1 : mat.col, col_off = transposed ? mat.row : 1;
        for (int j = 0; j < col; j++)
            for (int i = 0; i < mat.row; i++)
                new_buf[size + i + j*row] = mat.m[i*row_off + j*col_off];

        if (m != buff) std::free(m);
        m = new_buf;
        col += mat.col;
        size += mat.size;
        transposed = true;
        
        return *this;
    }

    Matrix hstack(const std::vector<Matrix>& mats)
    {
        if (mats.empty()) return Matrix(0, 0);
        size_t row = mats[0].row;
        size_t col = 0;
        for (auto& mat : mats)
        {
            if (mat.row != row)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            col += mat.col;
        }

        Matrix stack(row, col);
        stack.transpose();
        size_t off = 0;
        for (auto& mat : mats)
        {
            if (mat.transposed)
                for (size_t i = 0; i < row; i++)
                    for (size_t j = 0; j < mat.col; j++)
                        stack.m[i + j * row + off] = mat.m[i + j * row];
            else
                for (size_t i = 0; i < row; i++)
                    for (size_t j = 0; j < mat.col; j++)
                        stack.m[i + j * row + off] = mat.m[i * col + j];
            
            off += mat.col * row;
        }
        
        return stack;
    }

    void Matrix::setCol(int64_t c, const std::initializer_list<double> &lst)
    {
        if (lst.size() > row)
            throw std::length_error("Matrix column length not matching in Matrix::setCol");
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::setCol");

        if (transposed) 
            std::copy(lst.begin(), lst.end(), m + c * row);
        else
        {
            int i = 0; 
            for(auto n : lst)
                m[c + i++ * row] = n;
        }
            
    }

    void Matrix::setRow(int64_t r, const std::initializer_list<double> &lst)
    {
        if (lst.size() > col)
            throw std::length_error("Matrix row length not matching in Matrix::setRow");
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setRow");

        if (!transposed) 
            std::copy(lst.begin(), lst.end(), m + r * col);
        else
        {
            int i = 0; 
            for(auto n : lst)
                m[r + i++ * col] = n;
        }
            
    }

    Matrix Matrix::getCol(int64_t c)
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::getCol");
        Matrix v(row);
        if (transposed)
            memcpy(v.m, m + c * row, row);
        else
            for (size_t i = 0; i < row; i++)
                v.m[i] = m[c + i * col];
        return v;
    }

    Matrix Matrix::getRow(int64_t r)
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::getRow");
        Matrix v(col);
        if (transposed)
            for (size_t i = 0; i < col; i++)
                v.m[i] = m[r + i * row];
        else
            memcpy(v.m, m + r * col, col);
        std::swap(v.row, v.col);
        return v;
    }

    void Matrix::print()
    {
        std::cout << "[";
        for (int i = 0; i < this->row; i++)
        {
            for (int j = 0; j < this->col; j++)
            {
                double num;
                if (transposed)
                    num = m[j * this->row + i];
                else
                    num = m[i * this->col + j];
                std::cout << num << " ";
            }
            if (i < this->row - 1) std::cout << "\n ";
        }
        std::cout << "]\n";
    }

    Matrix Matrix::T()
    {
        return {*this, true};
    }

    void zero(Matrix& mat)
    {
        memset(mat.m, 0, mat.size * sizeof(double));
    }

    Matrix identity(size_t size)
    {
        Matrix I(size, size);
        zero(I);
        for (size_t i = 0; i < size; i++)
            I.m[i * size + i] = 1.0;
        return I;
    }

#endif


#ifdef ALGEBRA_SHORT_NAMES
    using Vec = Matrix;
    using Mat = Matrix;
#endif

}