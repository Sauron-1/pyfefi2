#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <compress/array.hpp>

namespace py = pybind11;

struct WriteBuffer {
    char* __restrict ptr;
    size_t offset = 0;

    WriteBuffer(char* ptr) : ptr(ptr) {}

    size_t write(const char* __restrict data, size_t size) {
        std::memcpy(static_cast<char*>(ptr) + offset, data, size);
        offset += size;
        return size;
    }
};

struct PyBufferReader {
    const py::buffer_info& info;
    size_t offset = 0;

    PyBufferReader(const py::buffer_info& b) : info(b) {}

    size_t read(char* __restrict dst, size_t size) {
        size_t total_bytes = info.size * info.itemsize;
        if (offset + size > total_bytes) throw std::out_of_range("Read past buffer end");
        std::memcpy(dst, static_cast<char*>(info.ptr) + offset, size);
        offset += size;
        return size;
    }
};

py::dtype to_py_dtype(DataType type) {
    switch(type) {
        case DataType::FLOAT32: return py::dtype::of<float>();
        case DataType::FLOAT64: return py::dtype::of<double>();
        case DataType::INT8: return py::dtype::of<int8_t>();
        case DataType::UINT8: return py::dtype::of<uint8_t>();
        case DataType::INT16: return py::dtype::of<int16_t>();
        case DataType::UINT16: return py::dtype::of<uint16_t>();
        case DataType::INT32: return py::dtype::of<int32_t>();
        case DataType::UINT32: return py::dtype::of<uint32_t>();
        case DataType::INT64: return py::dtype::of<int64_t>();
        case DataType::UINT64: return py::dtype::of<uint64_t>();
        default: throw std::runtime_error("Unknown underlying data type.");
    }
}

DataType from_py_dtype(const py::dtype& dt) {
    if (dt.is(py::dtype::of<float>()))    return DataType::FLOAT32;
    if (dt.is(py::dtype::of<double>()))   return DataType::FLOAT64;
    if (dt.is(py::dtype::of<int8_t>()))   return DataType::INT8;
    if (dt.is(py::dtype::of<uint8_t>()))  return DataType::UINT8;
    if (dt.is(py::dtype::of<int16_t>()))  return DataType::INT16;
    if (dt.is(py::dtype::of<uint16_t>())) return DataType::UINT16;
    if (dt.is(py::dtype::of<int32_t>()))  return DataType::INT32;
    if (dt.is(py::dtype::of<uint32_t>())) return DataType::UINT32;
    if (dt.is(py::dtype::of<int64_t>()))  return DataType::INT64;
    if (dt.is(py::dtype::of<uint64_t>())) return DataType::UINT64;
    throw std::runtime_error("Unsupported or unknown numpy dtype.");
}

struct PyCompressConfig {
    std::string err_mode;
    double rel_err,
           abs_err,
           l2_err,
           psnr_err;

    CompressConfig get() const {
        CompressConfig config;
        config.relErrorBound = rel_err;
        config.absErrorBound = abs_err;
        config.l2normErrorBound = l2_err;
        config.psnrErrorBound = psnr_err;

        if (err_mode == "abs")
            config.errorBoundMode = SZ3::EB_ABS;
        else if (err_mode == "rel")
            config.errorBoundMode = SZ3::EB_REL;
        else if (err_mode == "abs_and_rel")
            config.errorBoundMode = SZ3::EB_ABS_AND_REL;
        else if (err_mode == "abs_or_rel")
            config.errorBoundMode = SZ3::EB_ABS_OR_REL;
        else if (err_mode == "l2")
            config.errorBoundMode = SZ3::EB_L2NORM;
        else if (err_mode == "psnr")
            config.errorBoundMode = SZ3::EB_PSNR;
        else
            throw std::invalid_argument(
                    std::format("Error bound mode {} not supported.", err_mode));

        return config;
    }
};

template<typename T, size_t Dim>
py::bytes py_compress_impl(const py::buffer_info& buf, std::vector<size_t> block_shape, bool do_compress, const PyCompressConfig& cfg) {
    std::array<size_t, Dim> shape;
    std::array<size_t, Dim> block;
    std::array<size_t, Dim> strides;
    for (int i = 0; i < Dim; ++i) {
        shape[i] = buf.shape[i];
        block[i] = (i < block_shape.size()) ? block_shape[i] : 64;
        strides[i] = buf.strides[i] / sizeof(T);
    }

    auto config = cfg.get();
    std::vector<char> buffer(minimum_array_buf_size<T, Dim>(config, shape, block, do_compress));
    WriteBuffer writer(buffer.data());
    compress_array(
            config, static_cast<const T*>(buf.ptr), writer, shape, block, strides, do_compress);
    buffer.resize(writer.offset);
    return py::bytes(buffer.data(), writer.offset);
}

template<typename T>
py::bytes py_compress_impl_dispatch(const py::buffer_info& buf, std::vector<size_t> block_shape, bool do_compress, const PyCompressConfig& cfg) {
    if (buf.ndim == 3)
        return py_compress_impl<T, 3>(buf, block_shape, do_compress, cfg);
    if (buf.ndim == 2)
        return py_compress_impl<T, 2>(buf, block_shape, do_compress, cfg);
    if (buf.ndim == 1)
        return py_compress_impl<T, 1>(buf, block_shape, do_compress, cfg);
    throw std::invalid_argument("Unsupported dim: " + std::to_string(buf.ndim));
}

py::bytes py_compress(
        py::array arr, bool do_compress,
        const std::string& err_mode,
        double rel_err, double abs_err,
        double l2_err, double psnr_err) {
    std::vector<size_t> default_block_shape(arr.ndim());
    if (arr.ndim() == 3)
        for (auto& s : default_block_shape) s = 64;
    else if (arr.ndim() == 2)
        for (auto& s : default_block_shape) s = 128;
    else if (arr.ndim() == 1)
        for (auto& s : default_block_shape) s = 1024;
    else
        throw std::invalid_argument("Unsupported dim: " + std::to_string(arr.ndim()));

    PyCompressConfig cfg(err_mode, rel_err, abs_err, l2_err, psnr_err);

    DataType dt = from_py_dtype(arr.dtype());
    py::buffer_info buf = arr.request();
    switch (dt) {
        case DataType::FLOAT32:
            return py_compress_impl_dispatch<float>(buf, default_block_shape, do_compress, cfg);
        case DataType::FLOAT64:
            return py_compress_impl_dispatch<double>(buf, default_block_shape, do_compress, cfg);
        case DataType::UINT8:
            return py_compress_impl_dispatch<uint8_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::UINT16:
            return py_compress_impl_dispatch<uint16_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::UINT32:
            return py_compress_impl_dispatch<uint32_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::UINT64:
            return py_compress_impl_dispatch<uint64_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::INT8:
            return py_compress_impl_dispatch<int8_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::INT16:
            return py_compress_impl_dispatch<int16_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::INT32:
            return py_compress_impl_dispatch<int32_t>(buf, default_block_shape, do_compress, cfg);
        case DataType::INT64:
            return py_compress_impl_dispatch<int64_t>(buf, default_block_shape, do_compress, cfg);
        default: throw std::runtime_error("Unknown underlying data type.");
    }
}

template<typename T, size_t Dim>
py::array py_decompress_impl(const py::buffer_info& buf, const std::array<size_t, Dim>& shape) {
    py::array_t<T> result(shape);

    std::array<size_t, Dim> element_strides;
    element_strides[Dim-1] = 1;
    if constexpr (Dim >= 2) {
        for (int d = Dim-2; d >= 0; --d) {
            element_strides[d] = element_strides[d+1] * shape[d+1];
        }
    }

    PyBufferReader reader(buf);
    decompress_array(reader, result.mutable_data(), shape, element_strides);

    return result;
}

template<typename T>
py::array py_decompress_impl_dispatch(const py::buffer_info& buf, const ArrayHeader& hdr) {
    if (hdr.dim == 3) {
        std::array<size_t, 3> shape;
        for (int i = 0; i < 3; ++i) shape[i] = hdr.shape[i];
        return py_decompress_impl<T, 3>(buf, shape);
    }
    else if (hdr.dim == 2) {
        std::array<size_t, 2> shape;
        for (int i = 0; i < 2; ++i) shape[i] = hdr.shape[i];
        return py_decompress_impl<T, 2>(buf, shape);
    }
    else {
        std::array<size_t, 1> shape;
        for (int i = 0; i < 1; ++i) shape[i] = hdr.shape[i];
        return py_decompress_impl<T, 1>(buf, shape);
    }
}

py::array py_decompress(const py::buffer& buf) {
    py::buffer_info info = buf.request();
    if (info.size < sizeof(ArrayHeader)) {
        throw std::invalid_argument("Buffer too small to contain header");
    }
    ArrayHeader hdr = *reinterpret_cast<const ArrayHeader*>(info.ptr);
    switch (hdr.type) {
        case DataType::FLOAT32:
            return py_decompress_impl_dispatch<float>(info, hdr);
        case DataType::FLOAT64:
            return py_decompress_impl_dispatch<double>(info, hdr);
        case DataType::UINT8:
            return py_decompress_impl_dispatch<uint8_t>(info, hdr);
        case DataType::UINT16:
            return py_decompress_impl_dispatch<uint16_t>(info, hdr);
        case DataType::UINT32:
            return py_decompress_impl_dispatch<uint32_t>(info, hdr);
        case DataType::UINT64:
            return py_decompress_impl_dispatch<uint64_t>(info, hdr);
        case DataType::INT8:
            return py_decompress_impl_dispatch<int8_t>(info, hdr);
        case DataType::INT16:
            return py_decompress_impl_dispatch<int16_t>(info, hdr);
        case DataType::INT32:
            return py_decompress_impl_dispatch<int32_t>(info, hdr);
        case DataType::INT64:
            return py_decompress_impl_dispatch<int64_t>(info, hdr);
        default: throw std::runtime_error("Unknown underlying data type.");
    }
}

void init_comparray_functions(py::module_& m) {
    m
        .def("compress_array", &py_compress, "Compress 1-3D array",
                py::arg("arr"), py::arg("do_compress")=true,
                py::arg("err_mode") = "abs_and_rel",
                py::arg("rel_err") = 1e-3, py::arg("abs_err") = 1e-3,
                py::arg("l2_err") = 0, py::arg("psnr_err") = 0)
        .def("decompress_array", &py_decompress, "Decompress 1-3D array",
                py::arg("buf"))
        ;
}
