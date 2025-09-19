#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>

#pragma once

namespace py = pybind11;

template<typename Arr1, typename...Arrs>
inline char check_same_dtype(const Arr1& arr1, const Arrs&...arrs) {
    char dtype = arr1.dtype().char_();
    if ((... or (arrs.dtype().char_() != dtype)))
        throw std::runtime_error("Array dtype mismatch.");
    return dtype;
}

template<typename Arr1, typename...Arrs>
inline size_t check_same_dim(const Arr1& arr1, const Arrs&...arrs) {
    size_t dim = arr1.ndim();
    if ((... or (arrs.ndim() != dim)))
        throw std::runtime_error("Array dimensions mismatch.");
    return dim;
}

template<typename Arr1, typename...Arrs>
inline std::vector<size_t> check_same_shape(const Arr1& arr1, const Arrs&...arrs) {
    std::vector<size_t> shape(arr1.ndim());
    for (int i = 0; i < shape.size(); ++i) {
        shape[i] = arr1.shape(i);
        if ((... or (arrs.shape(i) != shape[i])))
            throw std::runtime_error("Array shape mismatch.");
    }
    return shape;
}

template<typename Arr1, typename...Arrs>
inline std::vector<size_t> check_same_strides(const Arr1& arr1, const Arrs&...arrs) {
    std::vector<size_t> strides(arr1.ndim());
    for (int i = 0; i < strides.size(); ++i) {
        strides[i] = arr1.strides(i);
        if ((... or (arrs.strides(i) != strides[i])))
            throw std::runtime_error("Array shape mismatch.");
    }
    return strides;
}
