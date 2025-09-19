#include <pybind11/pybind11.h>

#pragma once

namespace py = pybind11;

void init_sphere_mod_ext(py::module_& m);
void init_cartesian_mod_ext(py::module_& m);
