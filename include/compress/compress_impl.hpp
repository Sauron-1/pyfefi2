#include "compress.hpp"

template<typename T>
size_t compress(const CompressConfig& config, const T* data, char* cmpData, size_t cmpCap) {
    return SZ_compress(config, data, cmpData, cmpCap);
}

template<typename T>
void decompress(CompressConfig& config, const char* data, size_t cmpData, T* decData) {
    SZ_decompress(config, data, cmpData, decData);
}

template<typename T>
size_t minimum_buffer_size(const CompressConfig& config) {
    return SZ3::SZ_compress_size_bound<T>(config);
}

#define DECLARE_COMPRESS(TYPE) \
    template size_t compress<TYPE>(const CompressConfig&, const TYPE*, char*, size_t);

#define DECLARE_DECOMPRESS(TYPE) \
    template void decompress<TYPE>(CompressConfig&, const char*, size_t, TYPE*);

#define DECLARE_MIN_SIZE(TYPE) \
    template size_t minimum_buffer_size<TYPE>(const CompressConfig&);

#define DECLARE(TYPE) \
    DECLARE_COMPRESS(TYPE)\
    DECLARE_DECOMPRESS(TYPE)\
    DECLARE_MIN_SIZE(TYPE)
