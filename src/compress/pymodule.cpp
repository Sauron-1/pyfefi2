#include <SZ3/utils/Config.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
#include <format>

#include <compress/compressed_file.hpp>

namespace py = pybind11;

class PyArrayProxy;

class PyCompressedFile {
    public:
        PyCompressedFile(
            const std::string& filename, 
            const std::string& err_mode,  // Default: 'abs'
            const std::string& mode,      // Default: 'r'
            size_t max_arrays,            // Default: 128
            double rel_err,               // Errors are all default to zero
            double abs_err,
            double l2_err,
            double psnr_err)
        {
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

            std::ios::openmode fmode = std::ios::in | std::ios::binary;
            if (mode == "w")
                fmode |= std::ios::out;
            else if (mode != "r")
                throw std::invalid_argument(
                        "File mode must be 'r' or 'w'.");

            cpp_repl = std::make_unique<CompressedFile>(filename, config, fmode, max_arrays);
        }

        void put_array(const std::string& name, py::array arr, bool compressed=true) {
            std::vector<size_t> default_block_shape(arr.ndim());
            if (arr.ndim() == 3)
                for (auto& s : default_block_shape) s = 64;
            else if (arr.ndim() == 2)
                for (auto& s : default_block_shape) s = 128;
            else if (arr.ndim() == 1)
                for (auto& s : default_block_shape) s = 1024;
            else
                throw std::invalid_argument("Unsupported dim: " + std::to_string(arr.ndim()));
            put_array(name, arr, default_block_shape, compressed);
        }

        void put_array(const std::string& name, py::array arr, std::vector<size_t> block_shape, bool compressed=true) {
            py::buffer_info buf = arr.request();

            if (buf.ndim == 1) {
                dispatch_type<1>(name, buf, block_shape, compressed);
            } else if (buf.ndim == 2) {
                dispatch_type<2>(name, buf, block_shape, compressed);
            } else if (buf.ndim == 3) {
                dispatch_type<3>(name, buf, block_shape, compressed);
            } else {
                throw std::runtime_error("Only 1D, 2D, and 3D arrays are currently supported.");
            }
        }

        bool has_array(const std::string& name) {
            return cpp_repl->has_array(name);
        }

        PyArrayProxy get_array(const std::string& name);

    private:
        std::unique_ptr<CompressedFile> cpp_repl;

        template<size_t Dim>
        void dispatch_type(const std::string& name, const py::buffer_info& buf, const std::vector<size_t>& block_shape, bool compressed) {
            if (buf.format == py::format_descriptor<float>::format()) do_put<float, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<double>::format()) do_put<double, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<int8_t>::format()) do_put<int8_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<uint8_t>::format()) do_put<uint8_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<int16_t>::format()) do_put<int16_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<uint16_t>::format()) do_put<uint16_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<int32_t>::format()) do_put<int32_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<uint32_t>::format()) do_put<uint32_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<int64_t>::format()) do_put<int64_t, Dim>(name, buf, block_shape, compressed);
            else if (buf.format == py::format_descriptor<uint64_t>::format()) do_put<uint64_t, Dim>(name, buf, block_shape, compressed);
            else throw std::runtime_error("Unsupported data type.");
        }

        template<typename T, size_t Dim>
        void do_put(const std::string& name, const py::buffer_info& buf, const std::vector<size_t>& block_shape, bool compressed) {
            std::array<size_t, Dim> shape;
            std::array<size_t, Dim> block;
            std::array<size_t, Dim> strides;
            for(size_t i = 0; i < Dim; ++i) {
                shape[i] = buf.shape[i];
                block[i] = (i < block_shape.size()) ? block_shape[i] : 64;
                strides[i] = buf.strides[i] / sizeof(T);
            }
            cpp_repl->put_array<T, Dim>(name, static_cast<const T*>(buf.ptr), shape, block, strides, compressed);
        }

        friend class PyArrayProxy;
};

class PyArrayProxy {
    public:
        PyArrayProxy(CompressedFile* file, const std::string& name) : file(file), name(name) {}

        py::array __getitem__(py::object index) {
            ArrayHeader hdr = file->read_array_header(name);
            std::vector<size_t> starts(hdr.dim, 0);
            std::vector<size_t> extends(hdr.dim, 0);
            py::list adj_indices; // Will hold the NumPy-compatible slice for the bounding box

            // Handle multi-dimensional slicing if a tuple is passed
            if (py::isinstance<py::tuple>(index)) {
                py::tuple tup = index.cast<py::tuple>();

                // First pass: count non-ellipsis elements and check for multiple ellipses
                size_t num_ellipsis = 0;
                for (size_t i = 0; i < tup.size(); ++i) {
                    if (py::isinstance<py::ellipsis>(tup[i])) {
                        num_ellipsis++;
                    }
                }
                if (num_ellipsis > 1) {
                    throw std::invalid_argument("an index can only have a single ellipsis ('...')");
                }

                size_t non_ellipsis_count = tup.size() - num_ellipsis;
                if (non_ellipsis_count > hdr.dim) {
                    throw std::out_of_range("Too many indices for array.");
                }

                size_t dim_idx = 0;
                for (size_t tup_idx = 0; tup_idx < tup.size(); ++tup_idx) {
                    py::object obj = tup[tup_idx];

                    if (py::isinstance<py::ellipsis>(obj)) {
                        size_t expand_count = hdr.dim - non_ellipsis_count;
                        for (size_t e = 0; e < expand_count; ++e) {
                            starts[dim_idx] = 0;
                            extends[dim_idx] = hdr.shape[dim_idx];
                            adj_indices.append(py::slice(0, hdr.shape[dim_idx], 1));
                            dim_idx++;
                        }
                    } else {
                        apply_slice(obj, hdr.shape[dim_idx], starts[dim_idx], extends[dim_idx], adj_indices);
                        dim_idx++;
                    }
                }

                // Implicit full slice for omitted trailing dimensions (if tuple was shorter and had no trailing ellipsis)
                while (dim_idx < hdr.dim) {
                    starts[dim_idx] = 0;
                    extends[dim_idx] = hdr.shape[dim_idx];
                    adj_indices.append(py::slice(0, hdr.shape[dim_idx], 1));
                    dim_idx++;
                }
            }
            // Handle standalone Ellipsis e.g., arr[...]
            else if (py::isinstance<py::ellipsis>(index)) {
                for (size_t i = 0; i < hdr.dim; ++i) {
                    starts[i] = 0;
                    extends[i] = hdr.shape[i];
                    adj_indices.append(py::slice(0, hdr.shape[i], 1));
                }
            }
            // Handle 1D slice or scalar index on the first dimension
            else {
                apply_slice(index, hdr.shape[0], starts[0], extends[0], adj_indices);
                for (size_t i = 1; i < hdr.dim; ++i) {
                    starts[i] = 0;
                    extends[i] = hdr.shape[i];
                    adj_indices.append(py::slice(0, hdr.shape[i], 1));
                }
            }

            py::dtype dt = get_dtype_from_enum(hdr.type);
            std::vector<size_t> temp_shape(extends.begin(), extends.begin() + hdr.dim);

            // Allocate the bounding box array
            py::array temp_arr(dt, temp_shape);

            // Calculate total volume to safely skip reading if the slice is mathematically empty (e.g., length 0)
            size_t volume = 1;
            for (size_t e : temp_shape) volume *= e;

            if (volume > 0) {
                file->get_array(name, temp_arr.mutable_data(), starts, extends);
            }

            // Apply the adjusted slice natively. Squeezing dims, striding, and negative indexing happen instantly here.
            return temp_arr[py::tuple(adj_indices)];
        }

        py::array get_all() {
            ArrayHeader hdr = file->read_array_header(name);
            py::dtype dt = get_dtype_from_enum(hdr.type);
            std::vector<size_t> shape(hdr.dim);
            for (size_t i = 0; i < hdr.dim; ++i) shape[i] = hdr.shape[i];

            py::array arr(dt, shape);
            file->get_array(name, arr.mutable_data());
            return arr;
        }

    private:
        CompressedFile* file;
        std::string name;

        void apply_slice(const py::object& obj, ssize_t dim_shape, size_t& start_out, size_t& extend_out, py::list& adj_indices) {
            if (py::isinstance<py::slice>(obj)) {
                ssize_t start, stop, step, slicelength;
                py::slice sl = obj.cast<py::slice>();

                if (!sl.compute(dim_shape, &start, &stop, &step, &slicelength)) {
                    throw py::error_already_set();
                }

                if (slicelength == 0) {
                    start_out = 0; extend_out = 0;
                    adj_indices.append(py::slice(0, 0, 1));
                } else {
                    ssize_t end_incl = start + (slicelength - 1) * step;
                    ssize_t min_idx = std::min(start, end_incl);
                    ssize_t max_idx = std::max(start, end_incl);

                    start_out = static_cast<size_t>(min_idx);
                    extend_out = static_cast<size_t>(max_idx - min_idx + 1);

                    ssize_t adj_start = start - min_idx;
                    py::object adj_stop;

                    if (step < 0) {
                        ssize_t expected_stop = adj_start + slicelength * step;
                        if (expected_stop < 0) adj_stop = py::none();
                        else adj_stop = py::int_(expected_stop);
                    } else {
                        ssize_t expected_stop = adj_start + slicelength * step;
                        adj_stop = py::int_(expected_stop);
                    }

                    adj_indices.append(py::slice(py::int_(adj_start), adj_stop, py::int_(step)));
                }
            } else if (py::isinstance<py::int_>(obj)) {
                ssize_t val = obj.cast<ssize_t>();
                if (val < 0) val += dim_shape;
                if (val < 0 || val >= dim_shape) throw std::out_of_range("Index out of bounds");

                start_out = static_cast<size_t>(val);
                extend_out = 1;
                adj_indices.append(0);
            } else {
                throw std::invalid_argument("Unsupported index type. Only slices, integers, and ellipsis are allowed.");
            }
        }

        py::dtype get_dtype_from_enum(DataType type) {
            switch(type) {
                case DataType::INT8: return py::dtype::of<int8_t>();
                case DataType::UINT8: return py::dtype::of<uint8_t>();
                case DataType::INT16: return py::dtype::of<int16_t>();
                case DataType::UINT16: return py::dtype::of<uint16_t>();
                case DataType::INT32: return py::dtype::of<int32_t>();
                case DataType::UINT32: return py::dtype::of<uint32_t>();
                case DataType::INT64: return py::dtype::of<int64_t>();
                case DataType::UINT64: return py::dtype::of<uint64_t>();
                case DataType::FLOAT32: return py::dtype::of<float>();
                case DataType::FLOAT64: return py::dtype::of<double>();
                default: throw std::runtime_error("Unknown underlying data type in file.");
            }
        }
};

PyArrayProxy PyCompressedFile::get_array(const std::string& name) {
    return PyArrayProxy(cpp_repl.get(), name);
}

void init_compress_ext(py::module_& m) {
    py::class_<PyArrayProxy>(m, "PyArrayProxy")
        .def("__getitem__", &PyArrayProxy::__getitem__)
        .def("get_all", &PyArrayProxy::get_all);

    py::class_<PyCompressedFile>(m, "CompressedFile")
        .def(py::init<const std::string&, const std::string&, const std::string&, size_t, double, double, double, double>(),
            py::arg("filename"), 
            py::arg("err_mode") = "abs", 
            py::arg("mode") = "r", 
            py::arg("max_arrays") = 128, 
            py::arg("rel_err") = 0.0, 
            py::arg("abs_err") = 0.0, 
            py::arg("l2_err") = 0.0, 
            py::arg("psnr_err") = 0.0)
        .def("put_array", static_cast<void (PyCompressedFile::*)(const std::string&, py::array, bool)>(&PyCompressedFile::put_array),
             "Put an array into the compressed file with default block sizes.", py::arg("name"), py::arg("arr"), py::arg("compressed") = true)
        .def("put_array", static_cast<void (PyCompressedFile::*)(const std::string&, py::array, std::vector<size_t>, bool)>(&PyCompressedFile::put_array),
             "Put an array into the compressed file with specified block shapes.", py::arg("name"), py::arg("arr"), py::arg("block_shape"), py::arg("compressed") = true)
        .def("get_array", &PyCompressedFile::get_array, py::arg("name"), "Returns a proxy object allowing for array slice extraction.")
        .def("__getitem__", &PyCompressedFile::get_array, py::arg("name"), "Returns a proxy object allowing for array slice extraction.")
        .def("has_array", &PyCompressedFile::has_array, py::arg("name"), "Check if array presents in the dataset")
        ;
}
