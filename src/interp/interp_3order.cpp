#include <interp/interp.hpp>
#include <interp/interp_ext.hpp>

void init_interp_ext_3order(py::module_& m) {
    m.def("interpf1d3", &interp<1, 3, float>);
    m.def("interpf2d3", &interp<2, 3, float>);
    m.def("interpf3d3", &interp<3, 3, float>);
    m.def("interpd1d3", &interp<1, 3, double>);
    m.def("interpd2d3", &interp<2, 3, double>);
    m.def("interpd3d3", &interp<3, 3, double>);
}
