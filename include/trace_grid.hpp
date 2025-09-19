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
            s = (s - start) / scale;
            e = (e - start) / scale;
            Float zero = 1e-10;

            fvec dir = e - s;
            auto len = std::sqrt(simd::reduce_add(dir*dir));
            if (len <= zero) return;
            dir = dir / len;
            fvec cur = s;
            fvec dir_sign = simd::select(
                    dir < -zero, -1,
                    simd::select(dir > zero, 1, 0));
            auto dir_is_zero = (dir_sign == 0);
            while (simd::reduce_add((e-cur)*dir) > -zero) {
                auto idx = to_idx1(cur);
                if (idx >= 0 || idx < m_size)
                    ptr[idx] = val;

                auto next_i = simd::floor(cur + dir_sign);
                auto t = simd::select(dir_is_zero, 1e20, (next_i - cur) / dir);
                Float min_t = simd::reduce_min(t) + Float(1e-6);
                cur += dir * min_t;
            }
        }

        void set_lines(T val, std::vector<fvec> line, std::array<Int, 2> skips=std::array<Int, 2>{0, 0}) {
            for (auto i = skips[0]; i < line.size()-skips[1]-1; ++i) {
                set_line(val, line[i], line[i+1]);
            }
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
            auto idx = simd::to_int(simd::floor(pos));
            return simd::reduce_add(idx*strides);
        }

};
