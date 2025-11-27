#include <vector>
#include <array>
#include <simd.hpp>

#pragma once

template<typename T, typename Float, size_t N>
class TraceGrid {
    public:
        static constexpr size_t dim = N;
        using Int = simd::int_of_t<Float>;
        using fvec = simd::vec<Float, N>;
        using ivec = simd::vec<Int, N>;

        TraceGrid(T* ptr_,
                fvec start_,
                fvec scale_,
                ivec shape_,
                ivec strides_) :
            ptr(ptr_), start(start_), scale(scale_), shape(shape_), strides(strides_)
        {
            start -= scale * 0.5;
            m_size = simd::reduce_prod(shape);
        }

        int to_idx(const fvec& pos) const { return to_idx1((pos - start) / scale); }

        fvec i2pos(int idx) const {
            ivec ipos;
            for (auto i  = 0; i < N; ++i) {
                ipos[N-i-1] = idx % shape[N-i-1];
                idx /= shape[N-i-1];
            }
            return (simd::to_float(ipos) + 0.5) * scale + start;
        }

        size_t size() const { return m_size; }

        template<typename...Idx>
            requires(sizeof...(Idx) == N)
        T& operator()(Idx...idx) {
            return ptr[to_idx(fvec(idx...))];
        }
        template<typename...Idx>
            requires(sizeof...(Idx) == N)
        T operator()(Idx...idx) const {
            return ptr[to_idx(fvec(idx...))];
        }

        template<typename Idx>
        T& operator()(std::array<Idx, N> idx) {
            return ptr[to_idx(idx)];
        }
        template<typename Idx>
        T operator()(std::array<Idx, N> idx) const {
            return ptr[to_idx(idx)];
        }

        void set_line(T val, fvec s, fvec e) {
            s = (s - start) / scale + Float(0.5);
            e = (e - start) / scale + Float(0.5);
            Float zero = 1e-7;

            auto idx = to_idxi(simd::to_int(simd::floor(e)));
            ptr[idx] = val;

            fvec dir = e - s;
            auto len = std::sqrt(simd::reduce_add(dir * dir));
            if (len <= zero) return;

            dir = dir / len;

            auto dir_is_zero = simd::abs(dir) <= zero;

            fvec delta = s - simd::floor(s);
            ivec grid = simd::to_int(s - delta);

            fvec base = simd::select(dir >= 0, Float(1), Float(0));

            do {
                auto idx = to_idxi(grid);
                ptr[idx] = val;

                auto t = simd::select(dir_is_zero, Float(1e10), (base - delta) / dir);
                t = simd::select(t == 0, Float(0.9), t);

                Float min_t = simd::reduce_min(t);
                len -= min_t;

                delta += dir * min_t;
                fvec offset = simd::select(
                        delta >= 0,
                        simd::select(
                            delta >= 1,
                            -1, 0), 1);
                grid -= simd::to_int(offset);
                delta += offset;
            } while (len > zero);
        }

        void set_lines(T val, const std::vector<fvec>& line, std::array<Int, 2> skips=std::array<Int, 2>{0, 0}) {
            for (auto i = skips[0]; i+1 < line.size()-skips[1]; ++i)
                set_line(val, line[i], line[i+1]);
        }

        size_t get_total(T init_val) const {
            size_t total = 0;
            for (auto i = 0; i < m_size; ++i)
                if (ptr[i] != init_val) ++total;
            return total;
        }

    private:
        T *ptr;
        fvec start, scale;
        ivec shape, strides;
        size_t m_size;

        auto to_idx1(const fvec& pos) const {
            auto idx = simd::to_int(simd::round(pos));
            return simd::reduce_add(idx*strides);
        }
        auto to_idxi(const ivec& pos) const {
            return simd::reduce_add(pos*strides);
        }

};
