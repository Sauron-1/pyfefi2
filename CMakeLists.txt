cmake_minimum_required(VERSION 3.11)
project(pyfefi)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

if (NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

set(PYFEFI_ARCH "core-avx2" CACHE STRING "Architecture to tell gcc to optimize for (-march)")
message("-- Compiling for march=${PYFEFI_ARCH}")

include(FetchContent)

find_package(OpenMP REQUIRED)

find_package(pybind11 QUIET)
if (NOT pybind11_FOUND)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11/CMakeLists.txt")
        message(STATUS "pybind11 not found. Using local copy from thirdparty/pybind11.")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11")
    else()
        message(STATUS "pybind11 not found. Trying to download from GitHub.")
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v3.0.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

find_package(xsimd QUIET)
if (NOT xsimd_FOUND)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/xsimd/CMakeLists.txt")
        message(STATUS "xsimd not found. Using local copy from thirdparty/xsimd.")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/xsimd")
    else()
        message(STATUS "xsimd not found. Trying to download from GitHub.")
        FetchContent_Declare(
            xsimd
            GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
            GIT_TAG        13.2.0
        )
        FetchContent_MakeAvailable(xsimd)
    endif()
endif()

pybind11_add_module(pyfefi_kernel
    src/coords/cartesian_mod.cpp
    src/coords/sphere_mod.cpp
    src/interp/interp_1order.cpp
    src/interp/interp_2order.cpp
    src/interp/interp_3order.cpp
    src/interp/interp_4order.cpp
    src/trace_line.cpp
    src/module.cpp
)

target_link_libraries(pyfefi_kernel PUBLIC xsimd)
target_link_libraries(pyfefi_kernel PUBLIC OpenMP::OpenMP_CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(pyfefi_kernel PRIVATE /O2 /fp:fast /DNDEBUG /arch:${PYFEFI_ARCH})
else()
    target_compile_options(pyfefi_kernel PRIVATE -O3 -march=${PYFEFI_ARCH} -ffast-math -DNDEBUG)
endif()
target_compile_features(pyfefi_kernel PRIVATE cxx_std_20)
target_include_directories(pyfefi_kernel PRIVATE include)

if (DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(OUTDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set_target_properties(pyfefi_kernel PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUTDIR})
    set_target_properties(pyfefi_kernel PROPERTIES OUTPUT_NAME pyfefi_kernel)
endif()
