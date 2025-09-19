#include <simd.hpp>
#include <numbers>

#include <coords/coords.hpp>
#include <coords/coords_ext.hpp>

template<typename T, size_t N>
FORCE_INLINE simd::vec<T, N> fgrid(const simd::vec<T, N>& p) {
    return ((T)0.11 * p - simd::erf((p-(T)10.0)/(T)12.0) - simd::erf((p+(T)10.0)/(T)12.0)) * (T)(100.0*2.0/4.0/0.44);
}

template<typename T, size_t N>
FORCE_INLINE simd::vec<T, N> fgrid_diff(const simd::vec<T, N>& p) {
    auto e1 = (p - (T)10.0) / (T)12.0,
         e2 = (p + (T)10.0) / (T)12.0;
    return (T)(100.0*2.0/4.0/0.44) * ((T)0.11 - (T)(1.0/6.0)*simd::rsqrt(std::numbers::pi_v<T>) *
            (simd::exp(-e1*e1) + simd::exp(-e2*e2)));
}

template<typename T, size_t N>
FORCE_INLINE simd::vec<T, N> solve_grid_impl(const simd::vec<T, N>& x) {
    simd::vec<T, N> tol = 1e-6, p = 2;
    int max_step = 7;
    for (int i = 0; i < max_step; ++i) {
        auto xp = fgrid(p);
        auto delta = xp - x;
        if (simd::all(simd::abs(delta) < tol))
            break;
        auto diff = fgrid_diff(p);
        p -= delta / diff;
    }
    return p;
}

template<typename T>
class SphereMod : public Coordinates<T> {
    public:
        static constexpr size_t simd_width = simd::simd_width_v<T>;

        T solve_grid(T x) {
            return solve_grid_impl(simd::vec<T, 1>{x})[0];
        }

        void to_cartesian_impl(
                const T* __restrict p,
                const T* __restrict q,
                const T* __restrict w,
                T* __restrict x,
                T* __restrict y,
                T* __restrict z,
                size_t N) const noexcept override {
            size_t rem = N % simd_width;
#pragma omp parallel for schedule(guided)
            for (int64_t i = 0; i < N-rem; i += simd_width)
                to_cartesian_one<simd_width>(p+i, q+i, w+i, x+i, y+i, z+i);
            for (size_t i = N-rem; i < N; ++i)
                to_cartesian_one<1>(p+i, q+i, w+i, x+i, y+i, z+i);
        }

        void from_cartesian_impl(
                const T* __restrict x,
                const T* __restrict y,
                const T* __restrict z,
                T* __restrict p,
                T* __restrict q,
                T* __restrict w,
                size_t N) const noexcept override {
            size_t rem = N % simd_width;
#pragma omp parallel for schedule(guided)
            for (int64_t i = 0; i < N-rem; i += simd_width)
                from_cartesian_one<simd_width>(x+i, y+i, z+i, p+i, q+i, w+i);
            for (size_t i = N-rem; i < N; ++i)
                from_cartesian_one<1>(x+i, y+i, z+i, p+i, q+i, w+i);
        }

    private:
        template<size_t N>
        FORCE_INLINE void to_cartesian_one(
                const T* __restrict p,
                const T* __restrict q,
                const T* __restrict w,
                T* __restrict x,
                T* __restrict y,
                T* __restrict z) const {
            using vec_t = simd::vec<T, N>;
            auto _p = vec_t::loadu(p);
            auto _q = vec_t::loadu(q);
            auto _w = vec_t::loadu(w);
            auto r = fgrid(_p);
            auto theta = _q;
            auto phi = _w;
            auto rho = r * simd::sin(theta);
            auto _y = r * simd::cos(theta),
                 _x = rho * simd::cos(phi),
                 _z = -rho * simd::sin(phi);
            _x.storeu(x);
            _y.storeu(y);
            _z.storeu(z);
        }

        template<size_t N>
        FORCE_INLINE void from_cartesian_one(
                const T* __restrict x,
                const T* __restrict y,
                const T* __restrict z,
                T* __restrict p,
                T* __restrict q,
                T* __restrict w) const {
            using vec_t = simd::vec<T, N>;
            auto _x = vec_t::loadu(x);
            auto _y = vec_t::loadu(y);
            auto _z = vec_t::loadu(z);
            auto rho = simd::hypot(_x, _z);
            auto r = simd::hypot(rho, _y);
            auto _p = solve_grid_impl(r),
                 _q = simd::atan2(rho, _y),
                 _w = simd::atan2(-_z, _x);
            _p.storeu(p);
            _q.storeu(q);
            _w.storeu(w);
        }
};

void init_sphere_mod_ext(py::module_& m) {
    py::class_<SphereMod<double>>(m, "SphereMod")
        .def(py::init())
        .def("solve_grid", &SphereMod<double>::solve_grid)
        .def("to_cartesian", &SphereMod<double>::to_cartesian)
        .def("from_cartesian", &SphereMod<double>::from_cartesian);
    py::class_<SphereMod<float>>(m, "SphereModf")
        .def(py::init())
        .def("solve_grid", &SphereMod<float>::solve_grid)
        .def("to_cartesian", &SphereMod<float>::to_cartesian)
        .def("from_cartesian", &SphereMod<float>::from_cartesian);
}
