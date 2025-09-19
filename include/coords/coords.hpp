#include <numpy_util.hpp>

#pragma once

template<typename T>
class Coordinates {

    public:
        py::tuple to_cartesian(py::array_t<T> p, py::array_t<T> q, py::array_t<T> w) const {
            size_t size = p.size();
            size_t dim = check_same_dim(p, q, w);
            auto shape = check_same_shape(p, q, w);
            auto strides = check_same_strides(p, q, w);
            py::array_t<T> x(shape, strides),
                           y(shape, strides),
                           z(shape, strides);

            to_cartesian_impl(
                    p.data(), q.data(), w.data(),
                    x.mutable_data(), y.mutable_data(), z.mutable_data(),
                    size);

            return py::make_tuple(x, y, z);
        }

        py::tuple from_cartesian(py::array_t<T> x, py::array_t<T> y, py::array_t<T> z) const {
            size_t size = x.size();
            size_t dim = check_same_dim(x, y, z);
            auto shape = check_same_shape(x, y, z);
            auto strides = check_same_strides(x, y, z);
            py::array_t<T> p(shape, strides),
                           q(shape, strides),
                           w(shape, strides);

            from_cartesian_impl(
                    x.data(), y.data(), z.data(),
                    p.mutable_data(), q.mutable_data(), w.mutable_data(),
                    size);

            return py::make_tuple(p, q, w);
        }

    protected:
        virtual void to_cartesian_impl(
                const T* __restrict p,
                const T* __restrict q,
                const T* __restrict w,
                T* __restrict x,
                T* __restrict y,
                T* __restrict z,
                size_t N) const noexcept = 0;

        virtual void from_cartesian_impl(
                const T* __restrict x,
                const T* __restrict y,
                const T* __restrict z,
                T* __restrict p,
                T* __restrict q,
                T* __restrict w,
                size_t N) const noexcept = 0;
};
