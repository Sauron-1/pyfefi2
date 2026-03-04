cmake_minimum_required(VERSION 3.12)
project(pyfefi)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(PYBIND11_FINDPYTHON NEW)
if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

set(PYFEFI_ARCH "core-avx2" CACHE STRING "Architecture to tell gcc to optimize for (-march)")
message("-- Compiling for march=${PYFEFI_ARCH}")

include(FetchContent)

find_package(OpenMP REQUIRED)

function(find_or_build_package PACKAGE_NAME GITURL GITTAG)
    if (EXISTS "${CMAKE_SOURCE_DIR}/thirdparty/${PACKAGE_NAME}/CMakeLists.txt")
        add_subdirectory("${CMAKE_SOURCE_DIR}/thirdparty/${PACKAGE_NAME}")
    else()
        message(STATUS "${CMAKE_SOURCE_DIR}/thirdparty/${PACKAGE_NAME}/CMakeLists.txt not found. Trying to use system ${PACKAGE_NAME}.")
        find_package(${PACKAGE_NAME} QUIET)
        if (NOT ${PACKAGE_NAME}_FOUND)
            message(STATUS "System ${PACKAGE_NAME} not found. Trying to download from GitHub.")
            FetchContent_Declare(
                ${PACKAGE_NAME}
                GIT_REPOSITORY ${GITURL}
                GIT_TAG        ${GITTAG}
            )
            FetchContent_MakeAvailable(${PACKAGE_NAME})
        endif()
    endif()
endfunction()

set(BUILD_SZ3_BINARY OFF)

find_or_build_package(xsimd https://github.com/xtensor-stack/xsimd.git 14.0.0)
find_or_build_package(SZ3 https://github.com/szcompressor/SZ3.git v3.3.2)
find_or_build_package(pybind11 https://github.com/pybind/pybind11.git v3.0.1)

add_library(compress STATIC
    src/compress/compress_impl_uint64_t.cpp
    src/compress/compress_impl_int8_t.cpp
    src/compress/compress_impl_uint16_t.cpp
    src/compress/compress_impl_uint32_t.cpp
    src/compress/compress_impl_double.cpp
    src/compress/compress_impl_uint8_t.cpp
    src/compress/compress_impl_float.cpp
    src/compress/compress_impl_int32_t.cpp
    src/compress/compress_impl_int64_t.cpp
    src/compress/compress_impl_int16_t.cpp
    src/compress/compressed_file.cpp
)

if(NOT TARGET SZ3::SZ3)
    add_library(SZ3::SZ3 ALIAS SZ3)
endif()
target_link_libraries(compress PUBLIC zstd SZ3::SZ3)
target_compile_features(compress PUBLIC cxx_std_20)
target_include_directories(compress PUBLIC include)
set_target_properties(compress PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(pyfefi_kernel
    src/coords/cartesian_mod.cpp
    src/coords/sphere_mod.cpp
    src/compress/pymodule.cpp
    src/compress/array.cpp
    src/interp/interp_1order.cpp
    src/interp/interp_2order.cpp
    src/interp/interp_3order.cpp
    src/interp/interp_4order.cpp
    src/trace_line.cpp
    src/module.cpp
    src/py_utils.cpp
)

target_link_libraries(pyfefi_kernel PUBLIC xsimd OpenMP::OpenMP_CXX compress)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(compress PRIVATE /O2 /DNDEBUG /arch:${PYFEFI_ARCH})
    target_compile_options(pyfefi_kernel PRIVATE /O2 /fp:fast /DNDEBUG /arch:${PYFEFI_ARCH})
else()
    target_compile_options(compress PRIVATE -O3 -DNDEBUG -march=${PYFEFI_ARCH})
    target_compile_options(pyfefi_kernel PRIVATE -O3 -march=${PYFEFI_ARCH} -ffast-math -DNDEBUG)
endif()
target_compile_features(pyfefi_kernel PRIVATE cxx_std_20)
target_include_directories(pyfefi_kernel PRIVATE include)

if (DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(OUTDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set_target_properties(pyfefi_kernel PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUTDIR})
    set_target_properties(pyfefi_kernel PROPERTIES OUTPUT_NAME pyfefi_kernel)
endif()
