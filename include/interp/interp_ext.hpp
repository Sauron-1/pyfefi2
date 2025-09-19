#include <pybind11/pybind11.h>

#pragma once

namespace py = pybind11;

void init_interp_ext_1order(py::module_& m);
void init_interp_ext_2order(py::module_& m);
void init_interp_ext_3order(py::module_& m);
void init_interp_ext_4order(py::module_& m);
