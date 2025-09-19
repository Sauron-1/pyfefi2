#include "bspline_weights.hpp"
#include <simd.hpp>

#pragma once

namespace bspline_interp {

template<size_t order, typename Float, size_t N>
FORCE_INLINE constexpr auto apply_weights(const simd::vec<Float, N>& val) {
    std::array<simd::vec<Float, N>, order+1> result;
    simd::constexpr_for<0, order+1, 1>([val, &result] (auto I) {
        constexpr int i = decltype(I)::value;
        result[i] = bspline_weight<Float, order>[i][order];
        if constexpr (order > 0) {
            simd::constexpr_for<1, order+1, 1>([i, val, &result](auto J) {
                constexpr int j = order - decltype(J)::value;
                result[i] = result[i] * val + bspline_weight<Float, order>[i][j];
            });
        }
    });
    return result;
}

template<size_t D, size_t N>
consteval std::array<std::array<int, D>, powi(N, D)> cartesian_prod_impl() {
    std::array<std::array<int, D>, powi(N, D)> values;
    for (size_t d = 0; d < D; ++d) {
        auto nseg = powi(N, d);
        auto seg_len = powi(N, D-d-1);
        for (auto n = 0; n < N; ++n)
            for (auto s = 0; s < nseg; ++s)
                for (auto i = 0; i < seg_len; ++i)
                    values[(s*N+n)*seg_len + i][d] = n;
    }
    return values;
}
template<size_t D, size_t N>
static inline constexpr std::array<std::array<int, D>, powi(N, D)> cartesian_prod = cartesian_prod_impl<D, N>();

template<size_t D, size_t O, typename Float, size_t W>
class InterpWeights {
    public:
        static constexpr size_t simd_width = W;
        using float_t = Float;
        using int_t = simd::int_of_t<Float>;
        using fvec = simd::vec<float_t, simd_width>;
        using ivec = simd::vec<int_t, simd_width>;

        static constexpr size_t num_points = powi(O+1, D);

        FORCE_INLINE InterpWeights() = default;
        FORCE_INLINE InterpWeights(const std::array<fvec, D>& pos, const std::array<int_t, D>& strides) {
            init(pos, strides);
        }

        FORCE_INLINE void init(const std::array<fvec, D>& pos, const std::array<int_t, D>& strides) {
            std::array<std::array<fvec, O+1>, D> dim_weights;
            ivec idx_center = 0;
            simd::constexpr_for<0, D, 1>([=, &dim_weights, &idx_center](auto I) {
                constexpr int i = decltype(I)::value;
                fvec fidx;
                if constexpr (O % 2 == 0)
                    fidx = simd::round(pos[i]);
                else
                    fidx = simd::floor(pos[i]);
                dim_weights[i] = apply_weights<O>(pos[i] - fidx);
                idx_center += (simd::to_int(fidx) - (O/2)) * strides[i];
            });

            for (auto& w : m_weights) w = 1;

            constexpr auto cart_prod = cartesian_prod<D, O+1>;
            simd::constexpr_for<0, num_points, 1>([=, this](auto I) {
                constexpr int i = decltype(I)::value;
                ivec idx = idx_center;
                simd::constexpr_for<0, D, 1>([=, &idx, this] (auto J) {
                    constexpr int j = decltype(J)::value;
                    m_weights[i] *= dim_weights[j][cart_prod[i][j]];
                    idx += strides[j] * cart_prod[i][j];
                });
                m_indices[i] = idx;
            });
        }

        FORCE_INLINE static auto inside(
                const std::array<fvec, D>& pos,
                const std::array<int_t, D>& shape) {
            typename ivec::vec_bool_t result = true;
            constexpr size_t mhi = O - O/2;
            simd::constexpr_for<0, D, 1>([=, &result](auto I) {
                constexpr int i = decltype(I)::value;
                fvec fidx;
                if constexpr (O%2 == 0)
                    fidx = simd::round(pos[i]);
                else
                    fidx = simd::floor(pos[i]);
                auto idx = simd::to_int(fidx);
                result &= ((idx - O/2) >= 0) & (idx + mhi < shape[i]);
            });
            return simd::to_floatb(result);
        }

        FORCE_INLINE fvec weight(int i) const { return m_weights[i]; }
        FORCE_INLINE ivec index(int i) const { return m_indices[i]; }
        FORCE_INLINE static constexpr size_t size() { return num_points; }

    private:
        std::array<fvec, num_points> m_weights;
        std::array<ivec, num_points> m_indices;
};

template<size_t D, size_t O, typename Float, size_t W>
class Interpolator : InterpWeights<D, O, Float, W> {
    public:
        static constexpr size_t simd_width = W;
        using Super = InterpWeights<D, O, Float, W>;
        using float_t = Float;
        using int_t = simd::int_of_t<Float>;
        using fvec = simd::vec<float_t, simd_width>;
        using ivec = simd::vec<int_t, simd_width>;

        using Super::num_points;
        using Super::index;
        using Super::weight;

        FORCE_INLINE Interpolator() = default;
        FORCE_INLINE Interpolator(const std::array<fvec, D>& pos, const std::array<int_t, D>& strides) : Super(pos, strides) {}

        FORCE_INLINE fvec gather(const Float* __restrict src) const {
            fvec result = 0;
            simd::constexpr_for<0, num_points, 1>([=, &result, this](auto I) {
                constexpr int i = decltype(I)::value;
                result += fvec::gather(src, index(i)) * weight(i);
            });
            return result;
        }

        FORCE_INLINE fvec scatter(Float* __restrict dst, const fvec& val) const {
            simd::constexpr_for<0, num_points, 1>([=, this](auto I) {
                constexpr int i = decltype(I)::value;
                (fvec::gather(dst, index(i)) + val*weight(i)).scatter(dst, index(i));
            });
        }

};

} // namespace bspline_interp
