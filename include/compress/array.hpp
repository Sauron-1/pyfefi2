#pragma once

#include <cstring>
#include <omp.h>

#include "compress.hpp"

enum class DataType : uint8_t {
    INT8 = 0, UINT8, INT16, UINT16,
    INT32, UINT32, INT64, UINT64,
    FLOAT32, FLOAT64, UNKNOWN
};

template<typename T> DataType inline get_data_type() { return DataType::UNKNOWN; }
template<> DataType inline get_data_type<int8_t>() { return DataType::INT8; }
template<> DataType inline get_data_type<uint8_t>() { return DataType::UINT8; }
template<> DataType inline get_data_type<int16_t>() { return DataType::INT16; }
template<> DataType inline get_data_type<uint16_t>() { return DataType::UINT16; }
template<> DataType inline get_data_type<int32_t>() { return DataType::INT32; }
template<> DataType inline get_data_type<uint32_t>() { return DataType::UINT32; }
template<> DataType inline get_data_type<int64_t>() { return DataType::INT64; }
template<> DataType inline get_data_type<uint64_t>() { return DataType::UINT64; }
template<> DataType inline get_data_type<float>() { return DataType::FLOAT32; }
template<> DataType inline get_data_type<double>() { return DataType::FLOAT64; }

struct ArrayHeader {
    DataType type; // uint8_t
    uint8_t dim;
    uint8_t compressed;
    uint64_t shape[3] = {1, 1, 1};
    uint64_t block_size[3] = {1, 1, 1};
};

template<typename T, size_t Dim, typename OStream>
void compress_array(
        const CompressConfig& default_config,
        const T* __restrict data,
        OStream& out,
        const std::array<size_t, Dim>& shape,
        const std::array<size_t, Dim>& block_shape,
        const std::array<size_t, Dim>& strides,
        bool do_compress = true) {
    ArrayHeader arr_hdr;
    arr_hdr.type = get_data_type<T>();
    arr_hdr.dim = Dim;
    arr_hdr.compressed = do_compress ? 1 : 0;
    for (size_t i = 0; i < Dim; ++i) {
        arr_hdr.shape[i] = shape[i];
        arr_hdr.block_size[i] = block_shape[i];
    }

    std::array<size_t, Dim> grid;
    for (int i = 0; i < Dim; ++i)
        grid[i] = (shape[i] + block_shape[i] - 1) / block_shape[i];
    size_t num_blocks = 1;
    for (auto g : grid) num_blocks *= g;

    out.write(reinterpret_cast<char*>(&arr_hdr), sizeof(arr_hdr));
    std::vector<uint64_t> block_offsets(num_blocks + 1, 0);

    // Prepare workers
    int num_threads = omp_get_max_threads();
    size_t block_size = 1;
    for (auto s : block_shape) block_size *= s;
    CompressConfig config(default_config);
    config.setDims(block_shape.begin(), block_shape.end());
    size_t in_buf_size = block_size;

    std::vector<T> in_bufs(in_buf_size * num_threads);
    std::vector<CompressConfig> configs(num_threads, config);
    std::vector<std::vector<char>> outputs(num_blocks);
#pragma omp parallel for schedule(dynamic)
    for (size_t bid = 0; bid < num_blocks; ++bid) {
        int tid = omp_get_thread_num();
        T* __restrict in_buf = in_bufs.data() + in_buf_size*tid;
        CompressConfig& cfg = configs[tid];

        std::array<size_t, Dim> block_idx;
        size_t _bid = bid;
        for (int d = 0; d < Dim; ++d) {
            int _d = Dim-d-1;
            block_idx[_d] = _bid % grid[_d];
            _bid /= grid[_d];
        }

        std::array<size_t, Dim> start_idx;
        std::array<size_t, Dim> current_block_shape;
        for (size_t d = 0; d < Dim; ++d) {
            start_idx[d] = block_idx[d] * block_shape[d];
            current_block_shape[d] = std::min(block_shape[d], shape[d] - start_idx[d]);
        }
        cfg.setDims(current_block_shape.begin(), current_block_shape.end());

        if constexpr (Dim == 1) {
            for (size_t i = 0; i < current_block_shape[0]; ++i)
                in_buf[i] = data[(start_idx[0] + i) * strides[0]];
        } else if constexpr (Dim == 2) {
            for (size_t i = 0; i < current_block_shape[0]; ++i)
                for (size_t j = 0; j < current_block_shape[1]; ++j)
                    in_buf[i * current_block_shape[1] + j] =
                        data[(start_idx[0] + i) * strides[0] + (start_idx[1] + j) * strides[1]];
        } else if constexpr (Dim == 3) {
            for (size_t i = 0; i < current_block_shape[0]; ++i)
                for (size_t j = 0; j < current_block_shape[1]; ++j)
                    for (size_t k = 0; k < current_block_shape[2]; ++k)
                        in_buf[(i * current_block_shape[1] + j) * current_block_shape[2] + k] =
                            data[(start_idx[0] + i) * strides[0] + (start_idx[1] + j) * strides[1] + (start_idx[2] + k) * strides[2]];
        }

        size_t current_block_elements = 1;
        for (size_t d = 0; d < Dim; ++d) {
            current_block_elements *= current_block_shape[d];
        }

        if (do_compress) {
            size_t out_buf_size = minimum_buffer_size<T>(cfg);
            outputs[bid].resize(out_buf_size);
            size_t comp_size = compress(cfg, in_buf, outputs[bid].data(), out_buf_size);
            outputs[bid].resize(comp_size);
        } else {
            // Bypass compression: copy raw bytes directly into output block buffer
            size_t raw_bytes = current_block_elements * sizeof(T);
            outputs[bid].resize(raw_bytes);
            std::memcpy(outputs[bid].data(), in_buf, raw_bytes);
        }
    }

    // Finish header
    size_t size_acc = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        block_offsets[i] = size_acc;
        size_acc += outputs[i].size();
    }
    block_offsets[num_blocks] = size_acc;
    out.write(reinterpret_cast<char*>(block_offsets.data()), (num_blocks+1)*sizeof(uint64_t));

    // Write data
    for (size_t i = 0; i < num_blocks; ++i)
        out.write(reinterpret_cast<char*>(outputs[i].data()), outputs[i].size());
}

template<typename T, size_t Dim, typename IStream>
void decompress_array(
        IStream& data,
        T* __restrict dst,
        const std::array<size_t, Dim>& shape,
        const std::array<size_t, Dim>& strides) {

    // 1. Validate header against parameters (type and shape)
    ArrayHeader arr_hdr;
    if (!data.read(reinterpret_cast<char*>(&arr_hdr), sizeof(ArrayHeader))) {
        throw std::runtime_error("Failed to read ArrayHeader.");
    }

    if (arr_hdr.type != get_data_type<T>()) {
        throw std::runtime_error("Data type mismatch during decompression.");
    }
    if (arr_hdr.dim != Dim) {
        throw std::runtime_error("Dimension mismatch during decompression.");
    }
    for (size_t i = 0; i < Dim; ++i) {
        if (arr_hdr.shape[i] != shape[i]) {
            throw std::runtime_error("Shape mismatch during decompression.");
        }
    }

    // Calculate grid layout for block sizes
    std::array<size_t, Dim> grid;
    size_t num_blocks = 1;
    for (int i = 0; i < Dim; ++i) {
        grid[i] = (shape[i] + arr_hdr.block_size[i] - 1) / arr_hdr.block_size[i];
        num_blocks *= grid[i];
    }

    // Read block offsets to know chunk boundaries
    std::vector<uint64_t> block_offsets(num_blocks + 1);
    if (!data.read(reinterpret_cast<char*>(block_offsets.data()), (num_blocks + 1) * sizeof(uint64_t))) {
        throw std::runtime_error("Failed to read block offsets.");
    }

    // 2. Decompress in parallel (producer-consumer model)
    int num_threads = omp_get_max_threads();
    size_t max_block_elements = 1;
    for (size_t i = 0; i < Dim; ++i) {
        max_block_elements *= arr_hdr.block_size[i];
    }

    // Preallocate thread-local buffers to hold uncompressed block data
    std::vector<T> out_bufs(max_block_elements * num_threads);

    #pragma omp parallel
    {
        // The single thread acts as the Producer, reading chunks sequentially
        #pragma omp single
        {
            for (size_t bid = 0; bid < num_blocks; ++bid) {
                size_t comp_size = block_offsets[bid + 1] - block_offsets[bid];

                // Allocate dynamic buffer for this block's raw compressed bytes
                char* raw_comp_buf = new char[comp_size];
                data.read(raw_comp_buf, comp_size);

                // Consumer task: decompression and memory scattering
                #pragma omp task firstprivate(bid, raw_comp_buf, comp_size)
                {
                    int tid = omp_get_thread_num();
                    T* __restrict out_buf = out_bufs.data() + max_block_elements * tid;

                    // Reconstruct N-dimensional block index
                    std::array<size_t, Dim> block_idx;
                    size_t _bid = bid;
                    for (int d = 0; d < Dim; ++d) {
                        int _d = Dim - d - 1;
                        block_idx[_d] = _bid % grid[_d];
                        _bid /= grid[_d];
                    }

                    // Calculate start index and shape for the current block (handling boundaries)
                    std::array<size_t, Dim> start_idx;
                    std::array<size_t, Dim> current_block_shape;
                    size_t current_block_elements = 1;
                    for (size_t d = 0; d < Dim; ++d) {
                        start_idx[d] = block_idx[d] * arr_hdr.block_size[d];
                        current_block_shape[d] = std::min(static_cast<size_t>(arr_hdr.block_size[d]), shape[d] - start_idx[d]);
                        current_block_elements *= current_block_shape[d];
                    }

                    const T* source_buf = nullptr;

                    if (arr_hdr.compressed) {
                        CompressConfig cfg;
                        cfg.setDims(current_block_shape.begin(), current_block_shape.end());

                        // Assumed signature. Adjust to match your compress.hpp API:
                        // decompress(config, input_buffer, input_size, output_buffer, max_output_size)
                        decompress(cfg, raw_comp_buf, comp_size, out_buf);
                        source_buf = out_buf;
                    } else {
                        // If uncompressed, map the raw buffer directly
                        source_buf = reinterpret_cast<const T*>(raw_comp_buf);
                    }

                    // Scatter the block's data back into the strided destination array
                    if constexpr (Dim == 1) {
                        for (size_t i = 0; i < current_block_shape[0]; ++i)
                            dst[(start_idx[0] + i) * strides[0]] = source_buf[i];
                    } else if constexpr (Dim == 2) {
                        for (size_t i = 0; i < current_block_shape[0]; ++i)
                            for (size_t j = 0; j < current_block_shape[1]; ++j)
                                dst[(start_idx[0] + i) * strides[0] + (start_idx[1] + j) * strides[1]] =
                                    source_buf[i * current_block_shape[1] + j];
                    } else if constexpr (Dim == 3) {
                        for (size_t i = 0; i < current_block_shape[0]; ++i)
                            for (size_t j = 0; j < current_block_shape[1]; ++j)
                                for (size_t k = 0; k < current_block_shape[2]; ++k)
                                    dst[(start_idx[0] + i) * strides[0] + (start_idx[1] + j) * strides[1] + (start_idx[2] + k) * strides[2]] =
                                        source_buf[(i * current_block_shape[1] + j) * current_block_shape[2] + k];
                    }

                    // Clean up the dynamically allocated buffer for this task
                    delete[] raw_comp_buf;
                }
            }
        }
    }
}

template<typename T, size_t Dim>
size_t minimum_array_buf_size(
        const CompressConfig& default_config,
        const std::array<size_t, Dim>& shape,
        const std::array<size_t, Dim>& block_shape,
        bool do_compress = true) {

    // 1. ArrayHeader size
    size_t max_size = sizeof(ArrayHeader);

    // Compute grid layout
    std::array<size_t, Dim> grid;
    size_t num_blocks = 1;
    for (int i = 0; i < Dim; ++i) {
        grid[i] = (shape[i] + block_shape[i] - 1) / block_shape[i];
        num_blocks *= grid[i];
    }

    // 2. Block offsets array size
    max_size += (num_blocks + 1) * sizeof(uint64_t);

    // 3. Accumulate maximum buffer size needed for each block
    for (size_t bid = 0; bid < num_blocks; ++bid) {
        // Reconstruct N-dimensional block index
        std::array<size_t, Dim> block_idx;
        size_t _bid = bid;
        for (int d = 0; d < Dim; ++d) {
            int _d = Dim - d - 1;
            block_idx[_d] = _bid % grid[_d];
            _bid /= grid[_d];
        }

        // Calculate exact shape and elements for the current block (handling boundary blocks)
        std::array<size_t, Dim> start_idx;
        std::array<size_t, Dim> current_block_shape;
        size_t current_block_elements = 1;

        for (size_t d = 0; d < Dim; ++d) {
            start_idx[d] = block_idx[d] * block_shape[d];
            current_block_shape[d] = std::min(block_shape[d], shape[d] - start_idx[d]);
            current_block_elements *= current_block_shape[d];
        }

        if (do_compress) {
            CompressConfig cfg(default_config);
            cfg.setDims(current_block_shape.begin(), current_block_shape.end());
            max_size += minimum_buffer_size<T>(cfg);
        } else {
            max_size += current_block_elements * sizeof(T);
        }
    }

    return max_size;
}
