#include <SZ3/api/sz.hpp>

using CompressConfig = SZ3::Config;

template<typename T>
size_t compress(const CompressConfig& config, const T* data, char* cmpData, size_t cmpCap);

template<typename T>
void decompress(CompressConfig& config, const char* data, size_t cmpData, T* decData);

template<typename T>
size_t minimum_buffer_size(const CompressConfig& config);
