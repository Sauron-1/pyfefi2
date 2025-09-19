#include "rational.hpp"
#include <array>

#pragma once

namespace bspline_interp {

using namespace rational;
using std::size_t;

template<size_t order, size_t offset=0>
consteval auto bspline_weight_core() {
    std::array<std::array<Rational, order+1>, order+1> values;
    if constexpr (order == 0)
        values[0][0] = Rational(1);
    else {
        constexpr auto last0 = bspline_weight_core<order-1, offset>();
        constexpr auto last1 = bspline_weight_core<order-1, offset+1>();
        for (size_t i = 0; i < order; ++i) {
            for (size_t j = 0; j < order; ++j) {
                values[i][j+1] = values[i][j+1] + last0[i][j] / Rational(order);
                values[i+1][j+1] = values[i+1][j+1] - last1[i][j] / Rational(order);
                values[i+1][j] = values[i+1][j] + last1[i][j] * Rational(false, offset+order+1, order);
                values[i][j] = values[i][j] - last0[i][j] * Rational(false, offset, order);
            }
        }
    }
    return values;
}

inline size_t constexpr combination(size_t n, size_t r) {
    if (r == 0 or r == n) return 1;
    return combination(n, r-1) * (n-r+1) / r;
}

template<size_t order>
consteval auto converted_weights() {
    constexpr auto weights = bspline_weight_core<order>();
    std::array<std::array<Rational, order+1>, order+1> values;
    constexpr Rational delta = Rational(false, 1, 2-order%2);
    for (auto  i = 0; i < order+1; ++i)
        for (auto j = 0; j < order+1; ++j)
            for (auto k = 0; k <= j; ++k)
                values[i][k] = values[i][k] + combination(j, k) * weights[i][j] * pow(Rational(-1), k) * pow(delta+i, j-k);

    return values;
}

template<typename T, size_t order>
consteval auto bspline_weights_impl() {
    std::array<std::array<T, order+1>, order+1> weights;
    constexpr auto wr = converted_weights<order>();
    for (int i = 0; i < order+1; ++i)
        for (int j = 0; j < order+1; ++j)
            weights[i][j] = wr[i][j].template to_float<T>();
    return weights;
}

template<typename T, size_t order>
static inline constexpr auto bspline_weight = bspline_weights_impl<T, order>();

} // namespace bspline_interp
