#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <cstring>
#include <stdexcept>
#include <map>
#include <omp.h>

#include "compress.hpp"

enum class DataType : uint8_t {
    INT8 = 0, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64,
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

struct FileHeader {
    uint64_t flags = 0;
    uint32_t max_arrays;
    uint32_t num_arrays;
};

struct FileHeaderEntry {
    char name[24] = {0};
    uint64_t offset = 0;
};

struct ArrayHeader {
    DataType type; // uint8_t
    uint8_t dim;
    uint8_t compressed;
    uint64_t shape[3] = {1, 1, 1};
    uint64_t block_size[3] = {1, 1, 1};
};

class CompressedFile {
    public:

        CompressedFile(const std::string& filename, const CompressConfig& config, std::ios::openmode mode, size_t max_arrays = 128);

        ~CompressedFile() {
            if (m_file.is_open()) m_file.close();
        }

        template<typename T, size_t Dim>
        void put_array(
                const std::string& name,
                const T* data,
                const std::array<size_t, Dim>& shape,
                const std::array<size_t, Dim>& block_shape,
                const std::array<size_t, Dim>& strides,
                bool compressed = true);

        ArrayHeader read_array_header(const std::string& name) {
            uint64_t offset = m_entries.at(name);
            m_file.seekg(offset);
            ArrayHeader hdr;
            m_file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            return hdr;
        }

        DataType get_array(const std::string& name, void* buf, size_t cap=0) {
            ArrayHeader hdr = read_array_header(name);
            if (hdr.dim == 1)
                get_array_impl<1>(hdr, name, buf, cap);
            else if (hdr.dim == 2)
                get_array_impl<2>(hdr, name, buf, cap);
            else if (hdr.dim == 3)
                get_array_impl<3>(hdr, name, buf, cap);
            return hdr.type;
        }

        DataType get_array(const std::string& name, void* buf, std::vector<size_t> starts, std::vector<size_t> extends, size_t cap=0) {
            ArrayHeader hdr = read_array_header(name);
            if (hdr.dim == 1) {
                std::array<size_t, 1> s{starts[0]}, e{extends[0]};
                get_array_impl<1>(hdr, name, buf, s, e, cap);
            } else if (hdr.dim == 2) {
                std::array<size_t, 2> s{starts[0], starts[1]}, e{extends[0], extends[1]};
                get_array_impl<2>(hdr, name, buf, s, e, cap);
            } else if (hdr.dim == 3) {
                std::array<size_t, 3> s{starts[0], starts[1], starts[2]}, e{extends[0], extends[1], extends[2]};
                get_array_impl<3>(hdr, name, buf, s, e, cap);
            } else {
                throw std::runtime_error("Unsupported dimension");
            }
            return hdr.type;
        }

        bool has_array(const std::string& name) {
            return m_entries.contains(name);
        }

    private:
        std::string m_filename;
        std::fstream m_file;
        CompressConfig m_default_config;
        uint32_t m_max_arrays;
        std::map<std::string, uint64_t> m_entries;

        void load_header() {
            m_file.seekg(0);

            FileHeader header;
            m_file.read(reinterpret_cast<char*>(&header), sizeof(header));

            std::vector<FileHeaderEntry> entries(header.num_arrays);
            m_file.read(
                    reinterpret_cast<char*>(entries.data()),
                    sizeof(FileHeaderEntry)*header.num_arrays);
            for (const auto& entry : entries)
                m_entries[std::string(entry.name)] = entry.offset;

            m_max_arrays = header.max_arrays;
        }

        void write_header(bool init=false) {
            m_file.clear(); // Clear any existing stream errors to ensure writes succeed
            constexpr size_t entry_header_size = sizeof(FileHeaderEntry);

            FileHeader header{
                0,
                m_max_arrays,
                uint32_t(m_entries.size()) };

            std::vector<FileHeaderEntry> entries(m_max_arrays);
            size_t i = 0;
            for (const auto& [name, offset] : m_entries) {
                if (i >= m_max_arrays) break; // Guard against out-of-bounds
                std::strncpy(entries[i].name, name.c_str(), 24);
                entries[i].offset = offset;
                ++i;
            }

            m_file.seekp(0);
            m_file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            m_file.write(
                    reinterpret_cast<const char*>(entries.data()),
                    entries.size() * entry_header_size);
            m_file.flush(); // Force sync to disk immediately
        }

        template<size_t Dim>
        void get_array_impl(const ArrayHeader& hdr,
                const std::string& name, void* buf, size_t cap=0) {
            std::array<size_t, Dim> starts, extends;
            for (size_t i = 0; i < Dim; ++i) {
                starts[i] = 0;
                extends[i] = hdr.shape[i];
            }
            get_array_impl<Dim>(hdr, name, buf, starts, extends, cap);
        }

        template<size_t Dim>
        void get_array_impl(const ArrayHeader& hdr,
                const std::string& name, void* buf,
                std::array<size_t, Dim> starts, std::array<size_t, Dim> extends,
                size_t cap=0);
};
