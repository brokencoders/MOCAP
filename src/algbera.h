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
#include <cfloat>

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
        Matrix(size_t row, size_t col, const std::initializer_list<double> &lst, bool fill = true);
        Matrix(const Matrix &mat);
        Matrix(const Matrix &mat, bool transpose);
        Matrix(Matrix &&mat);
        ~Matrix();

        Matrix &operator=(const Matrix &m) &;
        Matrix &operator=(const Matrix &m) && = delete;

        Matrix  operator+(const Matrix &m) const &;
        Matrix& operator+(const Matrix &m) && { return operator+=(m); }
        Matrix  operator+(double s) const &;
        Matrix& operator+(double s) && { return operator+=(s); }
        Matrix& operator+=(const Matrix &m);
        Matrix& operator+=(double s);

        Matrix  operator-(const Matrix &m) const &;
        Matrix& operator-(const Matrix &m) && { return operator-=(m); }
        Matrix  operator-() const &;
        Matrix& operator-() &&;
        Matrix  operator-(double s) const &;
        Matrix& operator-(double s) && { return operator-=(s); }
        Matrix& operator-=(const Matrix &m);
        Matrix& operator-=(double s);

        Matrix  operator*(const Matrix &m) const;
        Matrix  operator*(double s) const &;
        Matrix& operator*(double s) && { return operator*=(s); }
        Matrix& operator*=(const Matrix &m);
        Matrix& operator*=(double s);

        Matrix  operator/(const Matrix& m) const &;
        Matrix& operator/(const Matrix& m) && { return operator/=(m); }
        Matrix  operator/(double s) const &;
        Matrix& operator/(double s) && { return operator/=(s); }
        Matrix& operator/=(const Matrix& m);
        Matrix& operator/=(double s);

        operator double() const;

        double norm() const;
        double normSquare() const;

        double max() const;
        double absMax() const;

        Matrix  hom() const &;
        Matrix& hom() &&;
        Matrix  hom_i() const &;
        Matrix& hom_i() &&;

        Matrix solve(Matrix b) const;
        Matrix cholesky() const;
        std::tuple<Matrix, Matrix> QR() const;
        std::tuple<Matrix, Matrix, Matrix> svd() const;

        Matrix& reshape(size_t row, size_t col);

        double &operator[](size_t i);
        double operator[](size_t i) const;
        double &val(size_t row, size_t col);
        Matrix& vstack(const Matrix& mat);
        Matrix& hstack(const Matrix& mat);

        void setCol(int64_t c, const std::initializer_list<double> &lst);
        void setCol(int64_t c, const Matrix& v);
        void setRow(int64_t r, const std::initializer_list<double> &lst);
        void setRow(int64_t r, const Matrix& v);
        void setSubMatrix(const Matrix& mat, int64_t r = 0, int64_t c = 0);
        Matrix  getCol(int64_t c) const &;
        Matrix& getCol(int64_t c) &&;
        Matrix  getRow(int64_t r) const &;
        Matrix& getRow(int64_t r) &&;
        Matrix  subMatrix(int64_t top, int64_t left, int64_t bottom = -1, int64_t right = -1) const &;

        void print() const;
        Matrix& transpose();
        Matrix T() const &;
        Matrix T() &&;

    private:
        static void realTranspose(const double* A, double* B, const size_t r, const size_t c, const size_t lda, const size_t ldb);
        static void realTransposeInPlace(double*& A, const size_t r, const size_t c, const size_t lda, bool local_buff);
        static Matrix householderReflect(const Matrix& u);
        void householderReflectSubMat(const Matrix& u, size_t r, size_t c);
        void householderReflectSubMatForward(const Matrix& u, size_t r, size_t c, bool store = true);

    private:
        double *m;
        size_t row, col, size;

        static const size_t buff_size = 16;
        alignas(32) double buff[buff_size];

    public:
        size_t rows() const { return row; }
        size_t cols() const { return col; }
        size_t getSize() const { return size; }

        friend void zero(Matrix&);
        friend Matrix  identity(size_t);
        friend Matrix  operator+(double, const Matrix&);
        friend Matrix& operator+(double, Matrix&&);
        friend Matrix  operator-(double, const Matrix&);
        friend Matrix& operator-(double, Matrix&&);
        friend Matrix  operator*(double, const Matrix&);
        friend Matrix& operator*(double, Matrix&&);
        friend double  operator/(double, const Matrix&);
        friend Matrix  vstack(const std::vector<Matrix>& mats);
        friend Matrix  hstack(const std::vector<Matrix>& mats);
    };

    void zero(Matrix& mat);
    Matrix identity(size_t size);

/* 
    class MatrixView
    {
    private:
        MatrixView(double* data, size_t row, size_t col, size_t row_offset)
            : m(data), row(row), col(col), roff(row_offset) {}

    public:
        MatrixView(const MatrixView&) = delete;
        MatrixView(const MatrixView&&) = delete;
        MatrixView& operator=(const MatrixView&) = delete;

    private:
        double *m;
        size_t row, col, roff;
        
        friend class Matrix;
    };
 */
    using Vector = Matrix;

#ifdef ALGEBRA_IMPL

    Matrix::Matrix(size_t length)
        : row(length), col(1), size(length)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
    }

    Matrix::Matrix(size_t row, size_t col)
        : row(row), col(col), size(row * col)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
    }

    Matrix::Matrix(const std::initializer_list<double> &lst)
        : row(lst.size()), col(1), size(lst.size())
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        std::copy(lst.begin(), lst.end(), m);
    }

    Matrix::Matrix(size_t row, size_t col, const std::initializer_list<double> &lst, bool fill)
        : row(row), col(col), size(row * col)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        std::copy(lst.begin(), std::min(lst.end(), lst.begin() + size), m);
        if (fill && lst.size() < size)
            std::fill(m + lst.size(), m + size, 0.0);
    }

    Matrix::Matrix(const Matrix &mat)
        : row(mat.row), col(mat.col), size(mat.size)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        std::copy(mat.m, mat.m + size, m);
        // std::cout << "Matrix copy constructor\n";
    }

    Matrix::Matrix(const Matrix &mat, bool)
        : row(mat.col), col(mat.row), size(mat.size)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = (double*)std::aligned_alloc(32, size * sizeof(double));
        if (row > 1 && col > 1)
            realTranspose(mat.m, m, mat.row, mat.col, mat.row, row);
        else
            std::copy(mat.m, mat.m + mat.size, m);
    }

    Matrix::Matrix(Matrix &&mat)
        : row(mat.row), col(mat.col), size(mat.size)
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
            std::copy(mat.m, mat.m + size, m);
        }
        // std::cout << "Matrix move constructor\n";
    }

    Matrix::~Matrix()
    {
        if (m != buff && m)
            std::free(m);
        // std::cout << "Matrix destructor\n";
    }

    Matrix &Matrix::operator=(const Matrix &mat) &
    {
        if (m != buff)
            std::free(m);
        if (mat.size > buff_size)
            m = (double*)std::aligned_alloc(32, mat.size * sizeof(double));
        else
            m = buff;

        row = mat.row, col = mat.col, size = mat.size;
        std::copy(mat.m, mat.m + size, m);
        return *this;
    }

    /* --------- SUM --------- */

    Matrix Matrix::operator+(const Matrix &m) const &
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        Matrix sum(row, col);
        for (size_t i = 0; i < size; i++)
            sum.m[i] = this->m[i] + m.m[i];
        return sum;
    }

    Matrix Matrix::operator+(double s) const &
    {
        Matrix sum(row, col);
        for (size_t i = 0; i < size; i++)
            sum.m[i] = m[i] + s;
        return sum;
    }

    Matrix operator+(double s, const Matrix& mat)
    {
        Matrix sum(mat.row, mat.col);
        for (size_t i = 0; i < mat.size; i++)
            sum.m[i] = mat.m[i] + s;
        return sum;
    }

    Matrix& operator+(double s, Matrix&& mat)
    {
        return mat.operator+=(s);
    }

    Matrix& Matrix::operator+=(const Matrix &m)
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        for (size_t i = 0; i < size; i++)
            this->m[i] += m.m[i];
        return *this;
    }

    Matrix& Matrix::operator+=(double s)
    {
        for (size_t i = 0; i < size; i++)
            m[i] += s;
        return *this;
    }

    /* --------- SUB --------- */

    Matrix Matrix::operator-(const Matrix &m) const &
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in sub operation");
        Matrix sub(row, col);
        for (size_t i = 0; i < size; i++)
            sub.m[i] = this->m[i] - m.m[i];
        return sub;
    }

    inline Matrix Matrix::operator-() const &
    {
        Matrix sub(row, col);
        for (size_t i = 0; i < size; i++)
            sub.m[i] = -m[i];
        return sub;
    }

    Matrix& Matrix::operator-() &&
    {
        for (size_t i = 0; i < size; i++)
            m[i] = -m[i];
        return *this;
    }

    Matrix operator-(double s, const Matrix& mat)
    {
        Matrix sub(mat.row, mat.col);
        for (size_t i = 0; i < mat.size; i++)
            sub.m[i] = mat.m[i] - s;
        return sub;
    }

    Matrix& operator-(double s, Matrix&& mat)
    {
        return mat.operator-=(s);
    }

    Matrix Matrix::operator-(double s) const &
    {
        Matrix sub(row, col);
        for (size_t i = 0; i < size; i++)
            sub.m[i] = m[i] - s;
        return sub;
    }

    Matrix& Matrix::operator-=(const Matrix &m)
    {
        if (row != m.row || col != m.col)
            throw std::length_error("Matrix sizes not matching in subtraction operation");
        for (size_t i = 0; i < size; i++)
            this->m[i] -= m.m[i];
        return *this;
    }

    Matrix& Matrix::operator-=(double s)
    {
        for (size_t i = 0; i < size; i++)
            m[i] -= s;
        return *this;
    }

    /* --------- MULT --------- */

    Matrix Matrix::operator*(const Matrix &mat) const
    {
        if (col != mat.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");
        Matrix mul(row, mat.col);
        double* t = (double*)std::aligned_alloc(32, size * sizeof(double));
        realTranspose(m, t, row, col, col, row);
        
        for (size_t i = 0; i < row; i++)
            for (size_t j = 0; j < mat.col; j++)
                mul.m[i * mul.col + j] = t[i] * mat.m[j];

        for (size_t k = 1; k < col; k++)
            for (size_t i = 0; i < row; i++)
            {
                auto& t_ik = t[i + k * row];
                for (size_t j = 0; j < mat.col; j++)
                    mul.m[i * mul.col + j] += t_ik * mat.m[k * mat.col + j];
            }
        
        std::free(t);
        return mul;
    }

    Matrix Matrix::operator*(double s) const &
    {
        Matrix mul(row, col);
        for (size_t i = 0; i < size; i++)
            mul.m[i] = m[i] * s;
        return mul;
    }

    Matrix operator*(double s, const Matrix& mat)
    {
        Matrix mul(mat.row, mat.col);
        for (size_t i = 0; i < mat.size; i++)
            mul.m[i] = mat.m[i] * s;
        return mul;
    }

    Matrix& operator*(double s, Matrix&& mat)
    {
        return mat.operator*=(s);
    }

    Matrix& Matrix::operator*=(const Matrix &mat)
    {
        if (col != mat.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");

        double *new_m;
        alignas(32) double tmp_buff[buff_size];
        if (row * mat.col <= buff_size)
        {
            new_m = buff;
            if (this->m == buff)
                realTranspose(m, tmp_buff, row, col, row, col);
            else
                realTransposeInPlace(m, row, col, col, m == buff);
        }
        else
        {
            new_m = (double*)std::aligned_alloc(32, row * mat.col * sizeof(double));
            realTransposeInPlace(m, row, col, col, m == buff);
        }

        for (size_t i = 0; i < row; i++)
            for (size_t j = 0; j < mat.col; j++)
                new_m[i * mat.col + j] = m[i] * mat.m[j];

        for (size_t k = 1; k < col; k++)
            for (size_t i = 0; i < row; i++)
            {
                auto& t_ik = m[i + k * row];
                for (size_t j = 0; j < mat.col; j++)
                    new_m[i * mat.col + j] += t_ik * mat.m[k * mat.col + j];
            }

        if (this->m != buff)
            free(this->m);
        this->m = new_m;
        col = mat.col;
        return *this;
    }

    Matrix& Matrix::operator*=(double s)
    {
        for (size_t i = 0; i < size; i++)
            m[i] *= s;
        return *this;
    }

    /* --------- DIV --------- */

    Matrix Matrix::operator/(const Matrix &mat) const &
    {
        if ((row != mat.row || col != mat.col) && (mat.col != 1 || mat.row != 1))
            throw std::length_error("Matrix sizes not matching in division operation");
        
        Matrix div(row, col);
        if (mat.col == 1 && mat.row == 1)
            for (size_t i = 0; i < size; i++)
                div.m[i] = m[i] / mat.m[0];
        else
            for (size_t i = 0; i < size; i++)
                div.m[i] = m[i] / mat.m[i];

        return div;
    }

    Matrix Matrix::operator/(double s) const &
    {
        Matrix div(row, col);
        for (size_t i = 0; i < size; i++)
            div.m[i] = m[i] / s;
        return div;
    }

    Matrix& Matrix::operator/=(const Matrix &mat)
    {
        if ((row != mat.row || col != mat.col) && (mat.col != 1 || mat.row != 1))
            throw std::length_error("Matrix sizes not matching in division operation");
        
        if (mat.col == 1 && mat.row == 1)
            for (size_t i = 0; i < size; i++)
                m[i] /= mat.m[0];
        else
            for (size_t i = 0; i < size; i++)
                m[i] /= mat.m[i];

        return *this;
    }

    Matrix& Matrix::operator/=(double s)
    {
        for (size_t i = 0; i < size; i++)
            m[i] /= s;
        return *this;
    }

    double operator/(double s, const Matrix& mat)
    {
        if (mat.col != 1 || mat.row != 1)
            throw std::length_error("Matrix is not 1x1 in scalar matrix division operation");
        return s / mat.m[0];
    }

    Matrix::operator double() const
    {
        if (row != 1 || col != 1)
            throw std::length_error("Matrix is not 1x1 in double cast");
        return m[0];
    }

    double Matrix::norm() const
    {
        return sqrt(normSquare());
    }

    double Matrix::normSquare() const
    {
        double sum = 0.;
        for (size_t i = 0; i < size; i++)
            sum += m[i] * m[i];
        return sum;
    }

    double Matrix::max() const
    {
        double max = m[0];
        for(int i = 0; i < size; i++)
            if(m[i] > max) max = m[i];
        return max;
    }


    double Matrix::absMax() const
    {
        double max = 0;
        for(int i = 0; i < size; i++)
            if(std::abs(m[i]) > max) max = std::abs(m[i]);
        return max;
    }


    Matrix Matrix::hom() const &
    {
        Matrix hv(size + 1);
        std::copy(m, m + size, hv.m);
        hv.m[size] = 1;
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    Matrix& Matrix::hom() &&
    {
        if (row != size && col != size)
            throw std::length_error("Only Vectors can be converted to homogeneus coordinates in Matrix::hom() &&");
        if (size + 1 <= buff_size)
        {
            if (m != buff)
            {
                std::copy(m, m + size, buff);
                std::free(m);
                m = buff;
            }
        }
        else
        {
            double* new_buf = (double*)std::aligned_alloc(32, size+1);
            std::copy(m, m + size, new_buf);
            std::free(m);
            m = new_buf;
        }
        m[size++] = 1.;
        if (row != 1) row++;
        else col++;
        return *this;
    }

    Matrix Matrix::hom_i() const &
    {
        Matrix hv(size - 1);
        for (size_t i = 0; i < hv.size; i++)
            hv.m[i] = m[i] / m[hv.size];
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    Matrix& Matrix::hom_i() &&
    {
        if (row != size && col != size)
            throw std::length_error("Only Vectors can be converted from homogeneus to cartesian coordinates in Matrix::hom_i() &&");
        double w = m[--size];
        for (size_t i = 0; i < size; i++)
            m[i] /= m[size];
        if (row != 1) row--;
        else col--;
        return *this;
    }

    Matrix Matrix::solve(Matrix b) const
    {
        Matrix x(col, b.col), R(*this);
        auto min = std::min(row-1, col);
        zero(x);

        for (size_t i = 0; i < min; i++)
        {
            Vector v = R.subMatrix(i,i, -1,i);
            v[0] += v[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v[0];
                R.householderReflectSubMatForward(v, i,i, false);
                b.householderReflectSubMat(v, i,0);
            }
        }
        for (int64_t i = x.row - 1; i >= 0; i--)
        {
            auto divisor = R.m[i*col + i];
            if (divisor > DBL_EPSILON || divisor < -DBL_EPSILON)
                x.setRow(i, (b.getRow(i) - R.getRow(i)*x)/divisor);
        }

        return x;
    }

    Matrix Matrix::cholesky() const
    {
        if (row != col)
            throw std::length_error("Matrix not square in Matrix::cholesky()");
        
        Matrix L(row, col);
        for (size_t i = 0; i < row; i++)
        {
            size_t j;
            for (j = 0; j < i; j++)
                L.m[i*col + j] = 0;
            for (; j < col; j++)
                L.m[i*col + j] = m[i*col + j];
        }

        for (size_t i = 0; i < L.col; i++)
        {
            double& a11 = L.m[i*col + i];
            a11 = sqrt(a11);
            if (std::isnan(a11))
                throw std::invalid_argument("Matrix not positive defined in Matrix::cholesky()");

            double a11_i = 1./a11;
            for (size_t j = i+1; j < row; j++)
                L.m[j + i*col] *= a11_i;

            for (size_t j = i+1; j < row; j++)
                for (size_t k = j; k < col; k++)
                    L.m[j*col + k] -= L.m[j + i*col] * L.m[k + i*col];
        }

        return L.T();
    }

    std::tuple<Matrix, Matrix> Matrix::QR() const
    {
        std::tuple<Matrix, Matrix> tpl = {identity(row), *this};
        auto& [Q, R] = tpl;
        auto min = std::min(row-1, col);

        for (size_t i = 0; i < min; i++)
        {
            Vector v = R.subMatrix(i,i, -1,i);
            v[0] += v[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v[0];
                R.householderReflectSubMatForward(v, i,i);
            }
        }
        for (int64_t i = min - 1; i >= 0; i--)
        {
            Vector u = R.subMatrix(i,i, -1,i);
            if (u.m[0] > DBL_EPSILON || u.m[0] < -DBL_EPSILON)
            {
                u.m[0] = 1.;
                Q.householderReflectSubMat(u, i,i);
            }
        }

        for (size_t i = 1; i < row; i++)
            std::fill(R.m + i*col, R.m + i*col + i, 0.);
        return tpl;
    }

    std::tuple<Matrix, Matrix, Matrix> Matrix::svd() const
    {
        return {Matrix(0), Matrix(0), Matrix(0)};
    }

    Matrix& Matrix::reshape(size_t row, size_t col)
    {
        if (size != row * col)
            throw std::length_error("Matrix wrong reshape size");
        this->row = row, this->col = col;
        return *this;
    }

    double &Matrix::operator[](size_t i)
    {
        return m[i];
    }

    double Matrix::operator[](size_t i) const
    {
        return m[i];
    }

    double &Matrix::val(size_t row, size_t col)
    {
        if (row >= this->row || col >= this->col)
            throw std::out_of_range("Matrix[] : index is out of range");
        return m[row * this->col + col];
    }

    Matrix& Matrix::vstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix columns not matching for vertical stacking");

        double* new_buf;

        if (col * (row + mat.row) < buff_size)
            new_buf = buff;
        else
            new_buf = (double*)std::aligned_alloc(32, col * (row + mat.row) * sizeof(double));
        
        if (new_buf != m)
            std::copy(m, m + size, new_buf);

        std::copy(mat.m, mat.m + mat.size, new_buf + size);

        if (m != buff && m != new_buf) std::free(m);
        m = new_buf;
        row += mat.row;
        size += mat.size;

        return *this;
    }

    Matrix vstack(const std::vector<Matrix>& mats)
    {
        size_t col = mats.empty() ? 0 : mats[0].col;
        size_t row = 0;
        for (auto& mat : mats)
        {
            if (mat.col != col)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            row += mat.row;
        }

        Matrix stack(row, col);
        double* buff = stack.m;
        for (auto& mat : mats)
        {
            std::copy(mat.m, mat.m + mat.size, buff);
            buff += mat.size;
        }
        
        return stack;
    }

    Matrix& Matrix::hstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix rows not matching for horizontal stacking");

        double* new_buf;

        if (row * (col + mat.col) < buff_size)
            new_buf = buff;
        else
            new_buf = (double*)std::aligned_alloc(32, row * (col + mat.col) * sizeof(double));
        
        if (new_buf == m)
            /* NOT parallelizable */
            for (size_t i = row-1; i > 0; i--)
                std::copy(m + i*col, m + (i+1)*col, new_buf + i*(col+mat.col));
        else
            for (size_t i = 0; i < row; i++)
                std::copy(m + i*col, m + (i+1)*col, new_buf + i*(col+mat.col));

        for (size_t i = 0; i < row; i++)
            std::copy(mat.m + i*mat.col, mat.m + (i+1)*mat.col, new_buf + i*(col+mat.col) + col);

        if (m != buff && m != new_buf) std::free(m);
        m = new_buf;
        col += mat.col;
        size += mat.size;
        
        return *this;
    }

    Matrix hstack(const std::vector<Matrix>& mats)
    {
        size_t row = mats.empty() ? 0 : mats[0].row;
        size_t col = 0;
        for (auto& mat : mats)
        {
            if (mat.row != row)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            col += mat.col;
        }

        Matrix stack(row, col);
        double* buff = stack.m;
        for (auto& mat : mats)
        {
            for (size_t i = 0; i < row; i++)
                std::copy(mat.m + i*mat.col, mat.m + (i+1)*mat.col, buff + i*col);

            buff += mat.col;
        }
        
        return stack;
    }

    void Matrix::setCol(int64_t c, const std::initializer_list<double> &lst)
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::setCol");

        size_t end = std::min(lst.size(), row);
        auto column = lst.begin();
        for (size_t i = 0; i < end; i++)
            m[c + i * row] = column[i];
    }

    void Matrix::setCol(int64_t c, const Matrix& v)
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::setCol");

        size_t end = std::min(v.getSize(), row);
        for (size_t i = 0; i < end; i++)
            m[c + i * row] = v[i];
    }

    void Matrix::setRow(int64_t r, const std::initializer_list<double> &lst)
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setRow");

        std::copy(lst.begin(), std::min(lst.end(), lst.begin() + col), m + r * col);
    }

    void Matrix::setRow(int64_t r, const Matrix& v)
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setRow");

        std::copy(v.m, v.m + std::min(v.getSize(), col), m + r * col);
    }

    void Matrix::setSubMatrix(const Matrix &mat, int64_t r, int64_t c)
    {
        if (c < 0) c += col;
        if (r < 0) r += row;
        if (r >= row || r < 0 || c >= col || c < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setSubMatrix");

        size_t endr = std::max(row - r, mat.row);
        size_t endc = std::max(col - c, mat.col);
        for (size_t i = 0; i < endr; i++)
            std::copy(mat.m + i*mat.col, mat.m + i*mat.col + endc, m + (r+i)*col + c);
    }

    Matrix Matrix::getCol(int64_t c) const &
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::getCol");
        Matrix v(row);
        for (size_t i = 0; i < row; i++)
            v.m[i] = m[c + i * col];
        return v;
    }

    Matrix& Matrix::getCol(int64_t c) &&
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::getCol &&");
        if (row <= buff_size)
        {
            if (m == buff)
                for (size_t i = 0; i < row; i++)
                    buff[i] = m[c + i*col];
            else
            {
                for (size_t i = 0; i < row; i++)
                    buff[i] = m[c + i*col];
                std::free(m);
                m = buff;
            }
        }
        else if (col > 1)
        {
            double* new_buf = (double*)std::aligned_alloc(32, row);
            for (size_t i = 0; i < row; i++)
                new_buf[i] = m[c + i*col];
            std::free(m);
            m = new_buf;
        }

        col = 1;
        size = row;
        return *this;
    }

    Matrix Matrix::getRow(int64_t r) const &
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::getRow");
        Matrix v(col);
        std::copy(m + r*col, m + (r+1)*col, v.m);
        std::swap(v.row, v.col);
        return v;
    }

    Matrix& Matrix::getRow(int64_t r) &&
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::getRow &&");
        if (col <= buff_size)
        {
            if (m == buff)
            {
                if (r != 0) std::copy(m + r*col, m + (r+1)*col, buff);
            }
            else
            {
                std::copy(m + r*col, m + (r+1)*col, buff);
                std::free(m);
                m = buff;
            }
        }
        else if (row > 1)
        {
            double* new_buf = (double*)std::aligned_alloc(32, col);
            std::copy(m + r*col, m + (r+1)*col, new_buf);
            std::free(m);
            m = new_buf;
        }

        row = 1;
        size = col;
        return *this;
    }

    Matrix Matrix::subMatrix(int64_t t, int64_t l, int64_t b, int64_t r) const &
    {
        if (t < 0) t += row;
        if (l < 0) l += col;
        if (b < 0) b += row;
        if (r < 0) r += col;
        if (l >= col || l < 0 || r >= col || r < 0 || t >= row || t < 0 || b >= row || b < 0)
            throw std::out_of_range("Matrix coordinates out of range in Matrix::subMatrix");
        if (t > b || l > r)
            throw std::out_of_range("Matrix coordinates negatively overlapping in Matrix::subMatrix &&");
        Matrix sm(b-t + 1, r-l + 1);
        for (size_t i = t; i <= b; i++)
            std::copy(m + i*col + l, m + i*col + r + 1, sm.m + (i-t)*sm.col);
        return sm;
    }

    void Matrix::print() const
    {
        std::cout << "[";
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                double num;
                num = m[i * this->col + j];
                std::cout << num << " ";
            }
            if (i < this->row - 1) std::cout << "\n ";
        }
        std::cout << "]\n";
    }

    Matrix& Matrix::transpose()
    {
        if (row > 1 && col > 1)
            realTransposeInPlace(m, row, col, col, m == buff);
        std::swap(row, col);
        return *this;
    }

    Matrix Matrix::T() const &
    {
        return Matrix(*this, true);
    }

    Matrix Matrix::T() &&
    {
        return transpose();
    }

    void Matrix::realTranspose(const double *A, double *B, const size_t r, const size_t c, const size_t lda, const size_t ldb)
    {
        const size_t block_size = 4UL;
        const size_t max_r = r & ~(block_size-1);
        const size_t max_c = c & ~(block_size-1);

        //#pragma omp parallel for
        for (size_t i = 0; i < max_r; i += block_size)
        {
            for (size_t j = 0; j < max_c; j += block_size)
                for (size_t k = 0; k < block_size; k++)
                    for (size_t l = 0; l < block_size; l++)
                        B[(j + l)*ldb + (i + k)] = A[(i + k)*lda + (j + l)];

            for (size_t k = 0; k < block_size; k++)
                for (size_t j = max_c; j < c; j++)
                    B[j*ldb + (i + k)] = A[(i + k)*lda + j];
        }

        for (size_t i = max_r; i < r; i++)
            for (size_t j = 0; j < c; j++)
                B[j*ldb + i] = A[i*lda + j];
    }

    void Matrix::realTransposeInPlace(double*& A, const size_t r, const size_t c, const size_t lda, bool lb)
    {
        if (lb)
        {
            alignas(32) double tmp_buff[buff_size];
            std::copy(A, A + r*c, tmp_buff);
            realTranspose(tmp_buff, A, r, c, lda, r);
        }
        else
        {
            double* tmp_buff = (double*)std::aligned_alloc(32, r*c * sizeof(double));
            realTranspose(A, tmp_buff, r, c, lda, r);
            std::free(A);
            A = tmp_buff;
        }
    }

    Matrix Matrix::householderReflect(const Matrix &u)
    {
        double tau_i = 2 / u.normSquare();
        Matrix ref = -tau_i*u*u.T();
        for (size_t i = 0; i < u.getSize(); i++)
            ref.val(i,i) += 1.;
        return ref;
    }

    void Matrix::householderReflectSubMat(const Matrix &u, size_t r, size_t c)
    {
        double tau_i = 2 / u.normSquare();
        Vector wt(1, col - c);

        for (size_t i = 0; i < wt.size; i++)
        {
            wt.m[i] = m[r*col + i + c];
            for (size_t k = 1; k < u.size; k++)
                wt.m[i] += u.m[k] * m[(r+k)*col + i + c];
            wt.m[i] *= tau_i;
        }
        for (size_t i = 0; i < wt.size; i++)
            m[r*col + c + i] -= wt.m[i];
        for (size_t i = 1; i < u.size; i++)
            for (size_t j = 0; j < wt.size; j++)
                m[(r+i)*col + c + j] -= u.m[i] * wt.m[j];
    }

    void Matrix::householderReflectSubMatForward(const Matrix &u, size_t r, size_t c, bool store)
    {
        double tau_i = 2 / u.normSquare(), theta = m[r*col + c];
        Vector wt(1, col - c - 1);

        for (size_t k = 1; k < u.size; k++)
            theta += u.m[k] * m[(r+k)*col + c];
        m[r*col + c] -= tau_i * theta;

        if (store)
            for (size_t i = 1; i < u.size; i++)
                m[(r+i)*col + c] = u.m[i];
        else
            for (size_t i = 1; i < u.size; i++)
                m[(r+i)*col + c] = 0;
        
        for (size_t i = 0; i < wt.size; i++)
        {
            wt.m[i] = m[r*col + i + c+1];
            for (size_t k = 1; k < u.size; k++)
                wt.m[i] += u.m[k] * m[(r+k)*col + i + c+1];
            wt.m[i] *= tau_i;
        }
        for (size_t i = 0; i < wt.size; i++)
            m[r*col + c+1 + i] -= wt.m[i];
        for (size_t i = 1; i < u.size; i++)
            for (size_t j = 0; j < wt.size; j++)
                m[(r+i)*col + c+1 + j] -= u.m[i] * wt.m[j];
    }

    void zero(Matrix& mat)
    {
        std::fill(mat.m, mat.m + mat.size, 0);
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