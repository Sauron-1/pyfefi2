#include <pybind11/pybind11.h>

#pragma once

namespace py = pybind11;

void init_trace_line_ext(py::module_& m);
