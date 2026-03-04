#include <compress/compressed_file.hpp>
#include <compress/array.hpp>

CompressedFile::CompressedFile(const std::string& filename, const CompressConfig& config, std::ios::openmode mode, size_t max_arrays):
    m_filename(filename), m_default_config(config), m_max_arrays(max_arrays)
{
    // Must be binary, readable file
    mode = mode | std::ios::binary | std::ios::in;
    m_file.open(m_filename, mode);

    if (!m_file.is_open()) {
        if (mode & std::ios::out) {
            m_file.open(filename, std::ios::out | std::ios::binary);
            write_header(true);
            m_file.close();

            mode &= ~std::ios::trunc; // Remove trunc so we don't wipe our fresh header
            m_file.open(filename, mode);
        }
        else {
            throw std::runtime_error("Cannot open file " + filename);
        }
    }

    // Validate the file size. If opened with trunc, it will be 0 bytes.
    m_file.seekg(0, std::ios::end);
    std::streamsize file_size = m_file.tellg();
    m_file.clear(); // Clear any EOF/fail flags set by tellg/seekg

    if (file_size == 0) {
        if (mode & std::ios::out) {
            write_header(true); // Initialize the file header safely
        } else {
            throw std::runtime_error("Cannot read from an empty file");
        }
    } else {
        load_header();
    }
}

template<typename T, size_t Dim>
void CompressedFile::put_array(
        const std::string& name,
        const T* data,
        const std::array<size_t, Dim>& shape,
        const std::array<size_t, Dim>& block_shape,
        const std::array<size_t, Dim>& strides,
        bool compressed) {
    static_assert(Dim <= 3, "Only 0-3 dimensional arrays are supported.");
    if (name.length() > 24) throw std::invalid_argument("Name must be within 24 characters.");
    if (m_entries.contains(name))
        throw std::runtime_error(
                std::format("Array {} already exists in file.", name));

    m_file.seekp(0, std::ios::end);
    uint64_t array_offset = m_file.tellp();
    m_entries[name] = array_offset;

    compress_array(m_default_config, data, m_file, shape, block_shape, strides, compressed);
    write_header(); // Fix: Update the file header so the array metadata persists
}

template<size_t Dim>
void CompressedFile::get_array_impl(const ArrayHeader& hdr,
        const std::string& name, void* buf,
        std::array<size_t, Dim> starts, std::array<size_t, Dim> extends,
        size_t cap) {
    // 1. check capacity if its not zero
    size_t num_elements = 1;
    for (size_t i = 0; i < Dim; ++i) num_elements *= extends[i];

    size_t element_size = 0;
    switch (hdr.type) {
        case DataType::INT8:   case DataType::UINT8:   element_size = 1; break;
        case DataType::INT16:  case DataType::UINT16:  element_size = 2; break;
        case DataType::INT32:  case DataType::UINT32:  case DataType::FLOAT32: element_size = 4; break;
        case DataType::INT64:  case DataType::UINT64:  case DataType::FLOAT64: element_size = 8; break;
        default: throw std::runtime_error("Unknown data type");
    }

    if (cap > 0 && cap < num_elements * element_size) {
        throw std::runtime_error("Insufficient capacity");
    }

    // 2. read necessary data into memory
    std::array<size_t, Dim> grid;
    size_t total_blocks = 1;
    for (size_t i = 0; i < Dim; ++i) {
        grid[i] = (hdr.shape[i] + hdr.block_size[i] - 1) / hdr.block_size[i];
        total_blocks *= grid[i];
    }

    uint64_t array_offset = m_entries.at(name);
    m_file.seekg(array_offset + sizeof(ArrayHeader));
    std::vector<uint64_t> block_offsets(total_blocks + 1);
    m_file.read(reinterpret_cast<char*>(block_offsets.data()), (total_blocks + 1) * sizeof(uint64_t));

    uint64_t data_start_offset = m_file.tellg();

    std::vector<size_t> blocks_to_process;
    for (size_t bid = 0; bid < total_blocks; ++bid) {
        std::array<size_t, Dim> block_idx;
        size_t _bid = bid;
        for (int d = 0; d < Dim; ++d) {
            int _d = Dim - d - 1;
            block_idx[_d] = _bid % grid[_d];
            _bid /= grid[_d];
        }

        bool overlap = true;
        for (size_t d = 0; d < Dim; ++d) {
            size_t b_start = block_idx[d] * hdr.block_size[d];
            size_t b_end = std::min(b_start + hdr.block_size[d], hdr.shape[d]);
            if (b_end <= starts[d] || b_start >= starts[d] + extends[d]) {
                overlap = false; break;
            }
        }
        if (overlap) blocks_to_process.push_back(bid);
    }

    std::vector<std::vector<char>> comp_data(blocks_to_process.size());
    for (size_t i = 0; i < blocks_to_process.size(); ++i) {
        size_t bid = blocks_to_process[i];
        size_t c_size = block_offsets[bid + 1] - block_offsets[bid];
        comp_data[i].resize(c_size);
        m_file.seekg(data_start_offset + block_offsets[bid]);
        m_file.read(comp_data[i].data(), c_size);
    }

    // 3. decompress in parallel, and write to buf
    auto dispatch_decompress = [&]<typename T>() {
        T* out_buf = reinterpret_cast<T*>(buf);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < blocks_to_process.size(); ++i) {
            size_t bid = blocks_to_process[i];

            std::array<size_t, Dim> block_idx;
            size_t _bid = bid;
            for (int d = 0; d < Dim; ++d) {
                int _d = Dim - d - 1;
                block_idx[_d] = _bid % grid[_d];
                _bid /= grid[_d];
            }

            std::array<size_t, Dim> b_start;
            std::array<size_t, Dim> current_block_shape;
            size_t block_elements = 1;
            for (size_t d = 0; d < Dim; ++d) {
                b_start[d] = block_idx[d] * hdr.block_size[d];
                current_block_shape[d] = std::min(hdr.block_size[d], hdr.shape[d] - b_start[d]);
                block_elements *= current_block_shape[d];
            }

            CompressConfig cfg(m_default_config);
            cfg.setDims(current_block_shape.begin(), current_block_shape.end());

            std::vector<T> dec_buf(block_elements);

            if (hdr.compressed) {
                decompress<T>(cfg, comp_data[i].data(), comp_data[i].size(), dec_buf.data());
            } else {
                std::memcpy(dec_buf.data(), comp_data[i].data(), comp_data[i].size());
            }

            if constexpr (Dim == 1) {
                size_t r_start = std::max(b_start[0], starts[0]);
                size_t r_end = std::min(b_start[0] + current_block_shape[0], starts[0] + extends[0]);
                for (size_t x = r_start; x < r_end; ++x)
                    out_buf[x - starts[0]] = dec_buf[x - b_start[0]];
            } else if constexpr (Dim == 2) {
                size_t r_start0 = std::max(b_start[0], starts[0]);
                size_t r_end0 = std::min(b_start[0] + current_block_shape[0], starts[0] + extends[0]);
                size_t r_start1 = std::max(b_start[1], starts[1]);
                size_t r_end1 = std::min(b_start[1] + current_block_shape[1], starts[1] + extends[1]);

                for (size_t x = r_start0; x < r_end0; ++x) {
                    for (size_t y = r_start1; y < r_end1; ++y) {
                        size_t out_idx = (x - starts[0]) * extends[1] + (y - starts[1]);
                        size_t in_idx = (x - b_start[0]) * current_block_shape[1] + (y - b_start[1]);
                        out_buf[out_idx] = dec_buf[in_idx];
                    }
                }
            } else if constexpr (Dim == 3) {
                size_t r_start0 = std::max(b_start[0], starts[0]);
                size_t r_end0 = std::min(b_start[0] + current_block_shape[0], starts[0] + extends[0]);
                size_t r_start1 = std::max(b_start[1], starts[1]);
                size_t r_end1 = std::min(b_start[1] + current_block_shape[1], starts[1] + extends[1]);
                size_t r_start2 = std::max(b_start[2], starts[2]);
                size_t r_end2 = std::min(b_start[2] + current_block_shape[2], starts[2] + extends[2]);

                for (size_t x = r_start0; x < r_end0; ++x) {
                    for (size_t y = r_start1; y < r_end1; ++y) {
                        for (size_t z = r_start2; z < r_end2; ++z) {
                            size_t out_idx = ((x - starts[0]) * extends[1] + (y - starts[1])) * extends[2] + (z - starts[2]);
                            size_t in_idx = ((x - b_start[0]) * current_block_shape[1] + (y - b_start[1])) * current_block_shape[2] + (z - b_start[2]);
                            out_buf[out_idx] = dec_buf[in_idx];
                        }
                    }
                }
            }
        }
    };

    switch (hdr.type) {
        case DataType::INT8:   dispatch_decompress.template operator()<int8_t>(); break;
        case DataType::UINT8:  dispatch_decompress.template operator()<uint8_t>(); break;
        case DataType::INT16:  dispatch_decompress.template operator()<int16_t>(); break;
        case DataType::UINT16: dispatch_decompress.template operator()<uint16_t>(); break;
        case DataType::INT32:  dispatch_decompress.template operator()<int32_t>(); break;
        case DataType::UINT32: dispatch_decompress.template operator()<uint32_t>(); break;
        case DataType::INT64:  dispatch_decompress.template operator()<int64_t>(); break;
        case DataType::UINT64: dispatch_decompress.template operator()<uint64_t>(); break;
        case DataType::FLOAT32: dispatch_decompress.template operator()<float>(); break;
        case DataType::FLOAT64: dispatch_decompress.template operator()<double>(); break;
        default: throw std::runtime_error("Unknown data type");
    }
}

#define DEFINE_PUT_ARRAY(TYPE, DIM) \
    template void CompressedFile::put_array<TYPE, DIM>( \
            const std::string&, const TYPE*, \
            const std::array<size_t, DIM>&, \
            const std::array<size_t, DIM>&, \
            const std::array<size_t, DIM>&, bool);

#define DEFINE_PUT_ARRAY_T(TYPE) \
    DEFINE_PUT_ARRAY(TYPE, 1) \
    DEFINE_PUT_ARRAY(TYPE, 2) \
    DEFINE_PUT_ARRAY(TYPE, 3) \

DEFINE_PUT_ARRAY_T(uint8_t);
DEFINE_PUT_ARRAY_T(uint16_t);
DEFINE_PUT_ARRAY_T(uint32_t);
DEFINE_PUT_ARRAY_T(uint64_t);

DEFINE_PUT_ARRAY_T(int8_t);
DEFINE_PUT_ARRAY_T(int16_t);
DEFINE_PUT_ARRAY_T(int32_t);
DEFINE_PUT_ARRAY_T(int64_t);

DEFINE_PUT_ARRAY_T(float);
DEFINE_PUT_ARRAY_T(double);

#define DEFINE_GET_ARRAY(DIM) \
    template void CompressedFile::get_array_impl( \
            const ArrayHeader&, \
            const std::string&, \
            void *, \
            std::array<size_t, DIM>, \
            std::array<size_t, DIM>, \
            size_t);

DEFINE_GET_ARRAY(1);
DEFINE_GET_ARRAY(2);
DEFINE_GET_ARRAY(3);
