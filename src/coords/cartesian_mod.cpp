#include <simd.hpp>
#include <coords/coords.hpp>
#include <coords/coords_ext.hpp>

template<typename T>
FORCE_INLINE T mytanh(T x, T x0, T d, T a1, T a2) {
    return (T)0.5 * ((a2 + a1) + (a2 - a1) * simd::tanh((x - x0) / d));
}

template<typename T, typename FnLeft, typename FnRight>
inline std::vector<T> integral(T dx0, T xlo, T xhi, FnLeft&& fnleft, FnRight&& fnright) {
    std::vector<T> xs(4000);

    size_t num_left = 0,
           idx = 0;

    T x = dx0,
      dx = dx0;

    for (int i = 0; i < 2000; ++i) {
        x -= dx;
        dx *= fnleft(x);
        xs[idx] = x;
        ++idx;
        ++num_left;
        if (x <= xlo - 4*dx) break;
    }

    x = 0;
    dx = dx0;
    for (int i = 0; i < 2000; ++i) {
        x += dx;
        dx *= fnright(x);
        xs[idx] = x;
        ++idx;
        if (x >= xhi + 4*dx) break;
    }

    std::vector<T> p2x(idx);
    for (int i = 0; i < num_left; ++i)
        p2x[i] = xs[num_left - i - 1];
    for (int i = num_left; i < idx; ++i)
        p2x[i] = xs[i];

    return p2x;
}

template<typename T>
class CMAxisEval {
    public:
        CMAxisEval(const T* p2x, const T* x2p, int len, int idx_min) :
            p2x(p2x), x2p(x2p), len(len), idx_min(idx_min) {}

        template<size_t N>
        FORCE_INLINE simd::vec<T, N> get(const simd::vec<T, N>& p) const {
            using vec_t = simd::vec<T, N>;
            auto fi = simd::floor(p);
            auto w1 = p - fi;
            auto i = simd::to_int(fi) + 4;
            auto p2x_i   = vec_t::gather(p2x, i  );
            auto p2x_ip1 = vec_t::gather(p2x, i+1);
            auto p2x_ip2 = vec_t::gather(p2x, i+2);
            auto p2x_im1 = vec_t::gather(p2x, i-1);
            return p2x_i +
                (p2x_ip1 - p2x_i) * w1 +
                T(0.5)*(p2x_ip1 - 2*p2x_i + p2x_im1) * (w1*w1) +
                T(1.0/6.0)*(p2x_ip2 - 3*p2x_ip1 + 3*p2x_i - p2x_im1) * (w1*w1*w1);
        }

        template<size_t N>
        FORCE_INLINE simd::vec<T, N> getr(const simd::vec<T, N>& x) const {
            using vec_t = simd::vec<T, N>;
            vec_t xlo(p2x[4]), xhi(p2x[len-5]);
            auto x1 = simd::clip(x, xlo, xhi) * 100;
            auto fi = simd::floor(x1);
            auto w1 = x1 - fi;
            auto i = simd::to_int(fi) - idx_min;
            auto x2p_i   = vec_t::gather(x2p, i  );
            auto x2p_ip1 = vec_t::gather(x2p, i+1);
            auto x2p_ip2 = vec_t::gather(x2p, i+2);
            auto x2p_im1 = vec_t::gather(x2p, i-1);
            return x2p_i +
                (x2p_ip1 - x2p_i) * w1 +
                T(0.5)*(x2p_ip1 - 2*x2p_i + x2p_im1) * (w1*w1) +
                T(1.0/6.0)*(x2p_ip2 - 3*x2p_ip1 + 3*x2p_i - x2p_im1) * (w1*w1*w1);
        }

    private:
        const T* __restrict p2x;
        const T* __restrict x2p;
        int len, idx_min;
};

template<typename T>
class CMAxis {
    public:
        CMAxis() = default;

        template<typename FnLeft, typename FnRight>
        CMAxis(T dx0, T xlo, T xhi, FnLeft&& fnleft, FnRight&& fnright) {
            init(dx0, xlo, xhi, std::forward<FnLeft>(fnleft), std::forward<FnRight>(fnright));
        }

        template<typename FnLeft, typename FnRight>
        void init(T dx0, T xlo, T xhi, FnLeft&& fnleft, FnRight&& fnright) {
            p2x = integral(dx0, xlo, xhi, std::forward<FnLeft>(fnleft), std::forward<FnRight>(fnright));
            int extra = std::max(int(dx0*20), 4);
            len = p2x.size();
            xlo = p2x[4];
            xhi = p2x[len-5];
            int idx_min = int(xlo*100) - extra;
            int idx_max = int(xhi*100) + extra;
            int num = idx_max - idx_min + 1;

            x2p.resize(num);
            for (auto& x : x2p) x = 0;

            int i0 = int(xlo*100) - 2;
            int i1 = int(xhi*100) + 2;
            int _j = 4;
            for (int i = i0; i < i1+1; ++i) {
                int idx = i - idx_min;
                T x = T(i) / 100.0;
                for (int j = _j; j < len; ++j) {
                    if (x < p2x[j]) {
                        _j = j - 1;
                        x2p[idx] = T(j-4) - (p2x[j] - x) / (p2x[j] - p2x[j-1]);
                        break;
                    }
                }
            }
            this->idx_min = idx_min;
        }

        size_t num_grids() const { return len - 8; }

        CMAxisEval<T> evaulator() const {
            return CMAxisEval<T>(p2x.data(), x2p.data(), len, idx_min);
        }

    private:
        std::vector<T> p2x, x2p;
        int idx_min, len;
};

template<typename T, size_t simd_width=simd::simd_width_v<T>>
class CartesianMod : public Coordinates<T> {
    public:
        using vec_t = simd::vec<T, simd_width>;
        using vec1_t = simd::vec<T, 1>;

        CartesianMod(std::array<T, 3> diff, std::array<std::array<T, 2>, 3> lims):
            xaxis(diff[0], lims[0][0], lims[0][1],
                    [](T x) { return mytanh<T>(x, -2, 1, 1.005, 1); },
                    [](T x) { return mytanh<T>(x, 1.2, 1, 1, 1.005); }),
            yaxis(diff[1], lims[1][0], lims[1][1],
                    [](T x) { return mytanh<T>(x, -1.5, 1, 1.01, 1); },
                    [](T x) { return mytanh<T>(x, 1.5, 1, 1, 1.01); }),
            zaxis(diff[2], lims[2][0], lims[2][1],
                    [](T x) { return mytanh<T>(x, -1.5, 1, 1.01, 1); },
                    [](T x) { return mytanh<T>(x, 1.5, 1, 1, 1.01); }) {}

        CartesianMod(
                std::array<T, 3> diff,
                std::array<std::array<T, 2>, 3> lims,
                std::array<std::array<T, 4>, 6> args):
            xaxis(diff[0], lims[0][0], lims[0][1],
                    [=](T x) { return mytanh<T>(x, args[0][0], args[0][1], args[0][2], args[0][3]); },
                    [=](T x) { return mytanh<T>(x, args[1][0], args[1][1], args[1][2], args[1][3]); }),
            yaxis(diff[1], lims[1][0], lims[1][1],
                    [=](T x) { return mytanh<T>(x, args[2][0], args[2][1], args[2][2], args[2][3]); },
                    [=](T x) { return mytanh<T>(x, args[3][0], args[3][1], args[3][2], args[3][3]); }),
            zaxis(diff[2], lims[2][0], lims[2][1],
                    [=](T x) { return mytanh<T>(x, args[4][0], args[4][1], args[4][2], args[4][3]); },
                    [=](T x) { return mytanh<T>(x, args[5][0], args[5][1], args[5][2], args[5][3]); }) {}

        std::array<size_t, 3> grid_size() const {
            std::array<size_t, 3> result;
            result[0] = xaxis.num_grids();
            result[1] = yaxis.num_grids();
            result[2] = zaxis.num_grids();
            return result;
        }

        void to_cartesian_impl(
                const T* __restrict p,
                const T* __restrict q,
                const T* __restrict w,
                T* __restrict x,
                T* __restrict y,
                T* __restrict z,
                size_t N) const noexcept override {
            auto xax = xaxis.evaulator();
            auto yax = yaxis.evaulator();
            auto zax = zaxis.evaulator();

            size_t rem = N % simd_width;
#pragma omp parallel for schedule(guided)
            for (int64_t i = 0; i < N-rem; i += simd_width) {
                xax.get(vec_t::loadu(p+i)).storeu(x+i);
                yax.get(vec_t::loadu(q+i)).storeu(y+i);
                zax.get(vec_t::loadu(w+i)).storeu(z+i);
            }

            for (size_t i = N-rem; i < N; ++i) {
                xax.get(vec1_t::loadu(p+i)).storeu(x+i);
                yax.get(vec1_t::loadu(q+i)).storeu(y+i);
                zax.get(vec1_t::loadu(w+i)).storeu(z+i);
            }
        }

        void from_cartesian_impl(
                const T* __restrict x,
                const T* __restrict y,
                const T* __restrict z,
                T* __restrict p,
                T* __restrict q,
                T* __restrict w,
                size_t N) const noexcept override {
            auto xax = xaxis.evaulator();
            auto yax = yaxis.evaulator();
            auto zax = zaxis.evaulator();
            size_t rem = N % simd_width;
#pragma omp parallel for schedule(guided)
            for (int64_t i = 0; i < N-rem; i += simd_width) {
                xax.getr(vec_t::loadu(x+i)).storeu(p+i);
                yax.getr(vec_t::loadu(y+i)).storeu(q+i);
                zax.getr(vec_t::loadu(z+i)).storeu(w+i);
            }

            for (size_t i = N-rem; i < N; ++i) {
                xax.getr(vec1_t::loadu(x+i)).storeu(p+i);
                yax.getr(vec1_t::loadu(y+i)).storeu(q+i);
                zax.getr(vec1_t::loadu(z+i)).storeu(w+i);
            }
        }

    private:
        CMAxis<T> xaxis, yaxis, zaxis;
};

void init_cartesian_mod_ext(py::module_& m) {
    py::class_<CartesianMod<double>>(m, "CartesianMod")
        .def(py::init<std::array<double, 3>, std::array<std::array<double, 2>, 3>>())
        .def(py::init<std::array<double, 3>, std::array<std::array<double, 2>, 3>, std::array<std::array<double, 4>, 6>>())
        .def("grid_sizes", &CartesianMod<double>::grid_size)
        .def("to_cartesian", &CartesianMod<double>::to_cartesian)
        .def("from_cartesian", &CartesianMod<double>::from_cartesian);
    py::class_<CartesianMod<float>>(m, "CartesianModf")
        .def(py::init<std::array<float, 3>, std::array<std::array<float, 2>, 3>>())
        .def(py::init<std::array<float, 3>, std::array<std::array<float, 2>, 3>, std::array<std::array<float, 4>, 6>>())
        .def("grid_sizes", &CartesianMod<float>::grid_size)
        .def("to_cartesian", &CartesianMod<float>::to_cartesian)
        .def("from_cartesian", &CartesianMod<float>::from_cartesian);
}
