#include <iostream>
#include <random>

#include <simd.hpp>
#include <rk.hpp>
#include <stdexcept>
#include <trace_grid.hpp>
#include <trace_line.hpp>
#include <numpy_util.hpp>
#include <interp/bspline_interp.hpp>
#include <omp.h>

#define TRACE_LINE_INTERP_ORDER 2

template<typename T>
struct TraceConfig {
    T step_size,
      tol,
      tol_rel,
      max_step,
      min_step,
      min_dist,
      term_val;
    int max_iter;

    void print() const {
        std::cout << "step_size: " << step_size << std::endl
                  << "tol: " << tol << std::endl
                  << "tol_rel: " << tol_rel << std::endl
                  << "max_step: " << max_step << std::endl
                  << "min_step: " << min_step << std::endl
                  << "min_dist: " << min_dist << std::endl
                  << "term_val: " << term_val << std::endl
                  << "max_iter: " << max_iter << std::endl;
    }
};

inline bool my_isnan(float val) {
    union { float f; uint32_t x; } u = {val};
    return (u.x << 1) > (0x7f800000u << 1);
}

inline bool my_isnan(double val) {
    union { double f; uint64_t x; } u = {val};
    return (u.x << 1) > (0x7ff0000000000000u << 1);
}

template<typename T, size_t N>
FORCE_INLINE constexpr simd::vec_bool<T, N> my_isnan(const simd::vec<T, N>& v) {
    return simd::array_map([](T val) { return my_isnan(val); }, v.arr);
}

template<typename T, size_t N>
class LineTracer {
    public:
        using Float = T;
        using Int = simd::int_of_t<Float>;
        using fvec = simd::vec<Float, N>;
        using ivec = simd::vec<Int, N>;

        template<typename...Arr>
            requires(sizeof...(Arr) == N)
        LineTracer(const py::array_t<Arr>&...arrs) :
            m_data{arrs.data()...}
        {
            delta = 1;
            start = 0;
            scale = std::sqrt(simd::reduce_add(delta*delta)) / Float(N);

            auto args = std::tie(arrs...);
            simd::constexpr_for<0, N, 1>([this, &args](auto I) {
                constexpr size_t i = decltype(I)::value;
                m_arr_refs[i] = std::get<i>(args);
            });

            const auto arr_tp = std::tie(arrs...);
            const auto& arr0 = std::get<0>(arr_tp);
            simd::constexpr_for<0, N, 1>([=, this](auto I) {
                constexpr int i = decltype(I)::value;
                if (std::get<i>(arr_tp).ndim() != N)
                    throw std::runtime_error("Dimension mismatch");
            });
            for (auto i = 0; i < N; ++i) {
                m_shape[i] = arr0.shape(i);
                m_strides[i] = arr0.strides(i) / sizeof(Float);
            }
            simd::constexpr_for<1, N, 1>([=, this](auto I) {
                constexpr int i = decltype(I)::value;
                for (int d = 0; d < N; ++d)
                    if (std::get<i>(arr_tp).strides(d) != m_strides[d]*sizeof(Float))
                        throw std::runtime_error("Stride mismatch");
            });
            m_size = simd::reduce_prod(m_shape);
        }

        template<typename...Arr>
            requires( sizeof...(Arr) == N + 2 )
        LineTracer(const py::array_t<Arr>&...arrs) {
            auto args = std::tie(arrs...);
            simd::constexpr_for<0, N, 1>([this, &args](auto I) {
                constexpr size_t i = decltype(I)::value;
                m_arr_refs[i] = std::get<i>(args);
                m_data[i] = std::get<i>(args).data();
            });

            simd::constexpr_for<0, N, 1>([=, this](auto I) {
                constexpr int i = decltype(I)::value;
                if (std::get<i>(args).ndim() != N)
                    throw std::runtime_error("Dimension mismatch");
            });

            for (auto i = 0; i < N; ++i) {
                delta[i] = std::get<N>(args).at(i);
                start[i] = std::get<N+1>(args).at(i);
            }
            scale = std::sqrt(simd::reduce_add(delta*delta)) / Float(N);

            const auto& arr0 = std::get<0>(args);
            for (auto i = 0; i < N; ++i) {
                m_shape[i] = arr0.shape(i);
                m_strides[i] = arr0.strides(i) / sizeof(Float);
            }
            simd::constexpr_for<1, N, 1>([=, this](auto I) {
                constexpr int i = decltype(I)::value;
                for (int d = 0; d < N; ++d)
                    if (std::get<i>(args).strides(d) != m_strides[d]*sizeof(Float))
                        throw std::runtime_error("Stride mismatch");
            });
            m_size = simd::reduce_prod(m_shape);
        }

        auto eval(const fvec& pos) const {
            auto coord = convert(pos);
            coord = clip(coord);
            std::array<simd::vec<Float, 1>, N> coord_for_interp;
            for (size_t d = 0; d < N; ++d)
                coord_for_interp[d] = coord[d];
            bspline_interp::Interpolator<N, TRACE_LINE_INTERP_ORDER, Float, 1> interp(coord_for_interp, m_strides.arr);
            fvec ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = interp.gather(m_data[i])[0];
            return ret;
        }

        auto operator()([[maybe_unused]] Float s, const fvec& coord) const {
            auto ret = eval(coord);
            Float norm = std::max(std::sqrt(simd::reduce_add(ret*ret)), Float(1e-10));
            return ret / norm;
        }

        template<bool neg>
        auto step(fvec& coord, TraceConfig<T> cfg) const {
            auto stepper = make_runge_kutta<T, N, 4>(*this);
            auto [result, real_step, err] = stepper.template step_adaptive<neg>(
                    0, coord, cfg.step_size, cfg.tol, cfg.tol_rel, cfg.max_step, cfg.min_step);
            return std::make_pair(result, real_step);
        }

        template<bool neg = false, typename TermFn>
        std::vector<fvec> trace(
                const fvec& init,
                TraceConfig<T> cfg,
                TermFn&& terminate) const {
            std::vector<fvec> result;
            result.push_back(init);
            while (not terminate(result.back()) and --cfg.max_iter > 0) {
                auto [res, real_step] = step<neg>(result.back(), cfg);
                if (simd::any(my_isnan(res))) break;
                cfg.step_size = real_step;
                result.push_back(res);
                if (cfg.min_dist > 0 and result.size() > 10) {
                    auto d = dist(init, result.back(), result[result.size()-2]);
                    if (d < cfg.min_dist)
                        break;
                }
            }
            return result;
        }

        template<typename TermFn>
        [[gnu::noinline]] std::vector<fvec> bidir_trace(
                const fvec& init,
                TraceConfig<T> cfg,
                TermFn&& terminate) const {
            auto res_neg = trace<true>(init, cfg, std::forward<TermFn>(terminate));
            auto res_pos = trace<false>(init, cfg, std::forward<TermFn>(terminate));
            size_t len_neg = res_neg.size(),
                   len_pos = res_pos.size();
            size_t length = len_neg + len_pos - 1;
            res_pos.resize(length);
            // move the positive part to the end of the vector
            if (len_neg == 1)
                return res_pos;
            std::move_backward(res_pos.begin(), res_pos.begin()+len_pos, res_pos.end());
            // reverse the negative part and copy them to the beginning of the vector
            std::reverse(res_neg.begin(), res_neg.end());
            std::copy(res_neg.begin(), res_neg.end(), res_pos.begin());
            return res_pos;
        }

        template<int dir=0>
        py::array trace_one_py(
                py::array_t<T, py::array::forcecast> init,
                Float step_size, Float tol, Float tol_rel, Float max_step, Float min_step, int max_iter, T min_dist, Float term_val) const {
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };
            fvec coords;
            for (auto i = 0; i < N; ++i)
                coords[i] = init.at(i);
            if constexpr (dir == 0) {
                auto result = bidir_trace(coords, cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                return to_numpy(result);
            }
            else if constexpr (dir == 1) {
                auto result = trace<false>(coords, cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                return to_numpy(result);
            }
            else if constexpr (dir == -1) {
                auto result = trace<true>(coords, cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                return to_numpy(result);
            }
        }

        py::list trace_many(
                py::array_t<T, py::array::forcecast> inits,
                Float step_size, Float tol, Float tol_rel, Float max_step, Float min_step, int max_iter, T min_dist, Float term_val) const {
            if (inits.ndim() != 2 or inits.shape(1) != N) {
                throw std::runtime_error("inits must be a (num_points, dim) array");
            }
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };
            size_t num_points = inits.shape(0);
            std::vector<fvec> coords(num_points);
            for (size_t i = 0; i < num_points; ++i)
                for (auto j = 0; j < N; ++j)
                    coords[i][j] = inits.at(i, j);

            std::vector<std::vector<fvec>> results(num_points);
#pragma omp parallel for schedule(guided)
            for (int64_t i = 0; i < num_points; ++i) {
                results[i] = bidir_trace(coords[i], cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
            }

            py::list res;
            for (auto& a : results) {
                res.append(to_numpy(a));
            }

            return res;
        }

        py::array find_roots(
                py::array_t<T, py::array::forcecast> inits,
                Float step_size, Float tol, Float tol_rel, Float max_step, Float min_step, int max_iter, T min_dist, Float term_val) const {
            if (inits.ndim() != 2 or inits.shape(1) != N) {
                throw std::runtime_error("inits must be a (num_points, dim) array");
            }
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };
            size_t num_points = inits.shape(0);
            std::vector<fvec> coords(num_points);
            for (auto i = 0; i < num_points; ++i)
                for (auto j = 0; j < N; ++j)
                    coords[i][j] = inits.at(i, j);

            std::array<size_t, 3> result_shape{ num_points, 4, N };
            py::array_t<T> result(result_shape);
            T* ptr = result.mutable_data();
#pragma omp parallel for schedule(dynamic) default(shared)
            for (int64_t i = 0; i < num_points; ++i) {
                auto pos = trace<false>(coords[i], cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                auto neg = trace<true>(coords[i], cfg, [this, term_val](auto& pos) { return terminate(pos, term_val); });
                for (auto j = 0; j < 2; ++j)
                    if (j < neg.size())
                        for (auto d = 0; d < N; ++d)
                            ptr[i*4*N + j*N + d] = neg[neg.size()-j-1][d];
                for (auto j = 0; j < 2; ++j)
                    if (j < pos.size())
                        for (auto d = 0; d < N; ++d)
                            ptr[i*4*N + (4-j-1)*N + d] = pos[pos.size()-j-1][d];
            }
            return result;
        }

        py::array find_open_close(
                py::array_t<uint8_t> ends_flags,
                py::array_t<uint8_t> trace_flags,
                Float step_size, Float tol, Float tol_rel, Float max_step, Float min_step, int max_iter, T min_dist, Float term_val,
                size_t report_num) const {
            constexpr uint16_t initial_flag = ~uint16_t(0);
            if (ends_flags.ndim() != N)
                throw std::invalid_argument("ends_flags must have dimension " + std::to_string(N));
            const std::array<Int, N> shape = m_shape.arr;
            for (auto i = 0u; i < N; ++i)
                if (ends_flags.shape(i) != shape[i])
                    throw std::invalid_argument("size of dimension " + std::to_string(i) + " of ends_flags does not match.");
            for (auto i = 0u; i < N; ++i)
                if (ends_flags.strides(i) != m_strides[i])
                    throw std::invalid_argument("Strides of dimension " + std::to_string(i) + " of ends_flags does not match.");
            for (auto i = 0u; i < N; ++i)
                if (trace_flags.strides(i) != m_strides[i])
                    throw std::invalid_argument("Strides of dimension " + std::to_string(i) + " of trace_flags does not match.");
            TraceConfig<T> cfg {
                .step_size = T(step_size),
                .tol = T(tol),
                .tol_rel = T(tol_rel),
                .max_step = T(max_step),
                .min_step = T(min_step),
                .min_dist = T(min_dist),
                .term_val = T(term_val),
                .max_iter = max_iter,
            };

            ivec result_strides;
            size_t current_stride = 1;
            for (int d = N - 1; d >= 0; --d) {
                result_strides[d] = current_stride;
                current_stride *= shape[d];
            }
            py::array_t<uint16_t> result(shape);
            uint16_t* __restrict result_ptr = result.mutable_data();
            TraceGrid<uint16_t, Float, N> trace_grid(
                    result_ptr,
                    start, delta, shape, result_strides);

            for (auto i = 0; i < result.size(); ++i)
                result_ptr[i] = initial_flag;
            const uint8_t* __restrict flags_ptr = ends_flags.data();
            const uint8_t* __restrict trace_flags_ptr = trace_flags.data();

            auto term_fn = [=, this](auto& pos) -> bool {
                bool term_zero = terminate(pos, term_val);
                if (term_zero)
                    return true;
                auto ipos = simd::to_int(convert(pos));
                auto i = simd::reduce_add(ipos * m_strides);
                return flags_ptr[i] > 0;
            };

            auto get_flag = [this, flags_ptr](auto& pos) -> uint16_t {
                auto ipos = simd::to_int(convert(pos));
                auto i = simd::reduce_add(ipos * m_strides);
                return flags_ptr[i];
            };

            ivec inner_shape = m_shape - 4;
            size_t total_points = simd::reduce_prod(inner_shape);
            size_t base_offset = simd::reduce_add(2 * m_strides);
            size_t base_r_offset = simd::reduce_add(2 * result_strides);

            std::vector<size_t> indices_sf(total_points);
            for (size_t i = 0; i < total_points; ++i) indices_sf[i] = i;
            std::shuffle(indices_sf.begin(), indices_sf.end(), std::mt19937(std::random_device()()));

            if (report_num == 0) report_num = total_points;

            std::vector<size_t> traced_num_thread(omp_get_max_threads());
            for (auto& tn : traced_num_thread) tn = 0;

            for (size_t seg = 0u; seg < total_points; seg += report_num) {
#pragma omp parallel for schedule(dynamic) default(shared)
                for (int64_t i = seg; i < seg+report_num; ++i) {
                    if (i >= total_points) continue;
                    auto idx = i2idx(indices_sf[i], inner_shape) + 2;
                    auto i_idx = simd::reduce_add(idx*m_strides);
                    auto r_idx = simd::reduce_add(idx*result_strides);
                    if (
                            result_ptr[r_idx] != initial_flag or // Already processed
                            trace_flags_ptr[i_idx] == 0 or  // User masked out
                            flags_ptr[i_idx] != 0 )  // Root region
                        continue;

                    int tid = omp_get_thread_num();
                    traced_num_thread[tid] += 1;

                    auto seed = simd::to_float(idx) * delta + start;
                    auto line = bidir_trace(seed, cfg, term_fn);
                    std::array<Int, 2> skips{0, 0};
                    size_t line_len = line.size();

                    if (line_len < 2)
                        continue;

                    while (skips[0] < line_len and not in_bound(convert(line[skips[0]])))
                        skips[0] += 1;
                    while (skips[1] < line_len and not in_bound(convert(line[line_len-skips[1]-1])))
                        skips[1] += 1;

                    if (skips[0]+skips[1] >= line_len)
                        continue;
                    uint16_t result_flag = (get_flag(line[skips[0]]) << 8) | get_flag(line[line_len-skips[1]-1]);

                    result_ptr[r_idx] = result_flag;
                    trace_grid.set_lines(
                            result_flag,
                            line, skips);
                }
                if (report_num < total_points) {
                    size_t traced_num = 0;
                    for (auto tn : traced_num_thread) traced_num += tn;
                    int print_len = int(std::log10(total_points)+1);
                    printf("%*ld / %*ld, %*ld points set, %*ld lines traced.\n",
                            print_len, std::min(seg+report_num, total_points),
                            print_len, total_points,
                            print_len, trace_grid.get_total(initial_flag),
                            print_len, traced_num);
                }
            }

            return result;
        }

        bool terminate(const fvec& coord, T term_val) const {
            auto coord_real = convert(coord);
            if (not in_bound(coord_real))
                return true;
            auto val = eval(coord);
            T norm = std::sqrt(simd::reduce_add(val*val));
            return norm <= term_val;
        }

        T dist(const fvec& init, const fvec& p1, const fvec& p2) const {
            auto diff1 = init - p1;
            auto diff2 = init - p2;
            auto diff = p1 - p2;
            auto d_i_p1_2 = simd::reduce_add(diff1*diff2);
            if (simd::reduce_add(diff*diff1) < 0 and simd::reduce_add(diff*diff2) > 0) {
                auto diff_norm = diff / std::max(std::sqrt(simd::reduce_add(diff*diff)), T(1e-10));
                auto project = std::abs(simd::reduce_add(diff_norm*diff1));
                return std::sqrt(d_i_p1_2 - project*project);
            }
            return std::sqrt(d_i_p1_2);
        }

    private:
        using Interp = bspline_interp::Interpolator<N, 2, T, 1>;
        fvec start, delta;
        ivec m_shape, m_strides;
        Float scale;
        size_t m_size;
        std::array<const Float*, N> m_data;
        std::array<py::object, N> m_arr_refs;

        FORCE_INLINE bool in_bound(const fvec& arr) const {
            constexpr Float buf = TRACE_LINE_INTERP_ORDER + 1;
            fvec lo = buf;
            fvec hi = simd::to_float(m_shape - buf - 1);
            return simd::all((arr >= lo) and (arr <= hi) and not my_isnan(arr));
        }

        FORCE_INLINE auto clip(const fvec& arr) const {
            constexpr Float buf = TRACE_LINE_INTERP_ORDER + 1;
            fvec lo = buf;
            fvec hi = simd::to_float(m_shape - buf - 1);
            return simd::clip(arr, lo, hi);
        }

        FORCE_INLINE auto convert(const fvec& coord) const {
            return (coord - start) / delta;
        }

        py::array to_numpy(const std::vector<fvec>& coords) const {
            std::array<size_t, 2> shape{ coords.size(), N };
            py::array_t<T> result(shape);
            T* ptr = result.mutable_data();
            for (auto i = 0; i < coords.size(); ++i)
                for (auto j = 0; j < N; ++j)
                    ptr[i*N + j] = coords[i][j];
            return result;
        }

        void copy_to_numpy(const std::vector<fvec>& coords, py::array_t<T>& out) const {
            std::vector<size_t> shape{coords.size(), N};
            out.resize(shape);
            auto out_uc = out.template mutable_unchecked<2>();
            for (auto i = 0; i < coords.size(); ++i)
                for (auto j = 0; j < N; ++j)
                    out_uc(i, j) = coords[i][j];
        }

        FORCE_INLINE static ivec i2idx(size_t i, const ivec& shape) {
            ivec result;
            for (size_t d = 0; d < N; ++d) {
                size_t _d = N - d - 1;
                result[_d] = i % shape[_d];
                i /= shape[_d];
            }
            return result;
        }
};

#define CFG_DEFAULTS \
    py::arg("step_size") = 0.01,  \
    py::arg("tol") = 1e-6,  \
    py::arg("tol_rel") = 1e-6,  \
    py::arg("max_step") = 1,  \
    py::arg("min_step") = 1e-4, \
    py::arg("max_iter") = 5000, \
    py::arg("min_dist") = 0.0, \
    py::arg("term_val") = 1e-2

#define DEFAULTS \
    py::arg("init"),  \
    CFG_DEFAULTS

using arg_t = py::array_t<float>;

void init_trace_line_ext(py::module_& m) {
    py::class_<LineTracer<float, 2>>(m, "LineTracer2")
        .def(py::init<const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"))
        .def(py::init<const arg_t, const arg_t, const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 2>::trace_one_py<0>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 2>::trace_many, DEFAULTS)
        .def("find_roots", &LineTracer<float, 2>::find_roots, DEFAULTS)
        .def("find_open_close", &LineTracer<float, 2>::find_open_close, py::arg("ends_flags"), py::arg("trace_flags"), CFG_DEFAULTS, py::arg("report_num")=0u)
        ;

    py::class_<LineTracer<float, 3>>(m, "LineTracer3")
        .def(py::init<const arg_t, const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"))
        .def(py::init<const arg_t, const arg_t, const arg_t,
                const arg_t, const arg_t>(),
                py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("delta"), py::arg("start"))
        .def("trace", &LineTracer<float, 3>::trace_one_py<0>, DEFAULTS)
        .def("trace_many", &LineTracer<float, 3>::trace_many, DEFAULTS)
        .def("find_roots", &LineTracer<float, 3>::find_roots, DEFAULTS)
        .def("find_open_close", &LineTracer<float, 3>::find_open_close, py::arg("ends_flags"), py::arg("trace_flags"), CFG_DEFAULTS, py::arg("report_num")=0u)
        ;
}
