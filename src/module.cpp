#include <pybind11/pybind11.h>

#include <coords/coords_ext.hpp>
#include <interp/interp_ext.hpp>
#include <trace_line.hpp>
#include <py_utils.hpp>

PYBIND11_MODULE(pyfefi_kernel, m) {
    auto coords_m = m.def_submodule("coords");
    init_sphere_mod_ext(coords_m);
    init_cartesian_mod_ext(coords_m);

    auto interp_m = m.def_submodule("interp");
    init_interp_ext_1order(interp_m);
    init_interp_ext_2order(interp_m);
    init_interp_ext_3order(interp_m);
    init_interp_ext_4order(interp_m);

    auto trace_m = m.def_submodule("tracer");
    init_trace_line_ext(trace_m);

    init_utils_ext(m);
}
