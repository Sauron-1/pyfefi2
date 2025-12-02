#include <py_utils.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

py::array quick_stack(const std::vector<py::array>& arrs) {
    int ndim = arrs[0].ndim();
    std::vector<size_t> result_shape(ndim + 1), result_strides(ndim+1);
    result_shape[0] = arrs.size();
    for (int i = 0; i < ndim; ++i)
        result_shape[i+1] = arrs[0].shape(i);
    result_strides[0] = arrs[0].size() * arrs[0].dtype().itemsize();
    for (int i = 0; i < ndim; ++i)
        result_strides[i+1] = arrs[0].strides(i);

    py::array result(arrs[0].dtype(), result_shape, result_strides);
    char* dst = result.mutable_unchecked<char>().mutable_data();
    size_t block_size = arrs[0].dtype().itemsize() * arrs[0].size();

    for (const auto& arr : arrs) {
        const char* src = arr.unchecked<char>().data();
#pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < block_size; ++i)
            dst[i] = src[i];
        dst += block_size;
    }
    return result;
}

void init_utils_ext(py::module_& m) {
    m.def("quick_stack", quick_stack);
}
