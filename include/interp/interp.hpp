#include <simd.hpp>
#include <interp/bspline_interp.hpp>
#include <numpy_util.hpp>
#include <stdexcept>

#pragma once

template<size_t dim, size_t order, typename Float>
auto interp_impl(
        const std::array<py::array_t<Float>, dim>& coords,
        const std::array<Float, dim>& start,
        const std::array<Float, dim>& dx,
        const std::vector<py::array_t<Float>>& srcs) {
    // Check source array shape and strides.
    using Int = simd::int_of_t<Float>;
    for (auto& src : srcs)
        if (src.ndim() != dim)
            throw std::runtime_error("Array have wrong dimension");
    std::array<Int, dim> shape, strides;
    for (int i = 0; i < dim; ++i) {
        shape[i] = srcs[0].shape(i);
        strides[i] = srcs[0].strides(i);
    }
    for (auto& src : srcs)
        for (int d = 0; d < dim; ++d) {
            if (shape[d] != src.shape(d))
                throw std::runtime_error("Shape mismatch");
            if (strides[d] != src.strides(d))
                throw std::runtime_error("Strides mismatch");
        }
    for (auto& s : strides) s /= sizeof(Float);

    // Check coords shape and strides
    size_t result_dim = coords[0].ndim();
    std::vector<size_t> result_shape(result_dim), result_strides(result_dim);
    for (int i = 0; i < result_dim; ++i) {
        result_shape[i] = coords[0].shape(i);
        result_strides[i] = coords[0].strides(i);
    }
    for (int d = 1; d < dim; ++d) {
        for (int i = 0; i < result_dim; ++i) {
            if (result_shape[i] != coords[d].shape(i))
                throw std::runtime_error("Coords shape mismatch");
            if (result_strides[i] != coords[d].strides(i))
                throw std::runtime_error("Coords strides mismatch");
        }
    }

    // Create result arrays
    std::vector<py::array_t<Float>> dsts;
    std::vector<Float*> dst_ptrs(srcs.size());
    for (int i = 0; i < srcs.size(); ++i) {
        dsts.emplace_back(result_shape, result_strides);
        dst_ptrs[i] = dsts.back().mutable_data();
    }

    // Extract pointers
    std::vector<const Float*> src_ptrs(srcs.size());
    for (int i = 0; i < srcs.size(); ++i) src_ptrs[i] = srcs[i].data();

    std::array<const Float*, dim> coords_ptrs;
    for (int d = 0; d < dim; ++d) coords_ptrs[d] = coords[d].data();

    // Real loop
    constexpr size_t simd_width = simd::simd_width_v<Float>;
    std::array<Float, dim> dx_inv, fill_pos;
    for (int d = 0; d < dim; ++d) dx_inv[d] = 1/dx[d];
    for (int d = 0; d < dim; ++d) fill_pos[d] = (Int)(order/2);
    size_t N = coords[0].size();
    size_t rem = N % simd_width;
    {
        using fvec = simd::vec<Float, simd_width>;
        using Interp = bspline_interp::InterpWeights<dim, order, Float, simd_width>;
        constexpr size_t num_points = Interp::size();
#pragma omp parallel for schedule(guided)
        for (int64_t i = 0; i < N-rem; i += simd_width) {
            std::array<fvec, dim> pos;
            simd::constexpr_for<0, dim, 1>([=, &pos](auto D) {
                constexpr int d = decltype(D)::value;
                pos[d] = (fvec::loadu(coords_ptrs[d] + i) - start[d]) * dx_inv[d];
            });
            //auto msk = inside<order>(shape, pos);
            auto msk = Interp::inside(pos, shape);
            simd::constexpr_for<0, dim, 1>([=, &pos](auto D) {
                constexpr int d = decltype(D)::value;
                pos[d] = simd::select(msk, pos[d], fill_pos[d]);
            });
            Interp interp(pos, strides);

            for (int s = 0; s < srcs.size(); ++s) {
                fvec result = 0;
                simd::constexpr_for<0, num_points, 1>([=, &result](auto J) {
                    constexpr int j = decltype(J)::value;
                    result += interp.weight(j) * fvec::gather(src_ptrs[s], interp.index(j));
                });
                result = simd::select(msk, result, std::nan(""));
                result.storeu(dst_ptrs[s] + i);
            }
        }
    }

    {
        using Interp = bspline_interp::InterpWeights<dim, order, Float, 1>;
        using fvec = simd::vec<Float, 1>;
        constexpr size_t num_points = Interp::size();
        for (size_t i = N-rem; i < N; ++i) {
            std::array<fvec, dim> pos;
            simd::constexpr_for<0, dim, 1>([=, &pos](auto D) {
                constexpr int d = decltype(D)::value;
                pos[d] = (fvec::loadu(coords_ptrs[d] + i) - start[d]) * dx_inv[d];
            });
            auto msk = Interp::inside(pos, shape);
            simd::constexpr_for<0, dim, 1>([=, &pos](auto D) {
                constexpr int d = decltype(D)::value;
                pos[d] = simd::select(msk, pos[d], fill_pos[d]);
            });
            Interp interp(pos, strides);

            for (int s = 0; s < srcs.size(); ++s) {
                fvec result = 0;
                simd::constexpr_for<0, num_points, 1>([=, &result](auto J) {
                    constexpr int j = decltype(J)::value;
                    result += interp.weight(j) * fvec::gather(src_ptrs[s], interp.index(j));
                });
                result = simd::select(msk, result, std::nan(""));
                result.storeu(dst_ptrs[s] + i);
            }
        }
    }

    return dsts;
}

template<size_t dim, size_t order, typename Float>
auto interp(
        const std::array<py::array_t<Float>, dim>& coords_to,
        const std::array<py::array_t<Float>, dim>& coords_from,
        const std::vector<py::array_t<Float>>& srcs) {
    for (auto c : coords_from)
        if (c.ndim() != 1)
            throw std::runtime_error("coords_from must be 1D arrays");
    std::array<Float, dim> start, dx;
    for (int d = 0; d < dim; ++d) {
        size_t N = coords_from[d].shape(0);
        start[d] = coords_from[d].at(0);
        dx[d] = (coords_from[d].at(N-1) - start[d]) / (N-1);
    }

    return interp_impl<dim, order, Float>(coords_to, start, dx, srcs);
}
