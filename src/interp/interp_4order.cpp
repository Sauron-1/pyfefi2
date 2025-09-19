#include <interp/interp.hpp>
#include <interp/interp_ext.hpp>

void init_interp_ext_4order(py::module_& m) {
    m.def("interpf1d4", &interp<1, 4, float>);
    m.def("interpf2d4", &interp<2, 4, float>);
    m.def("interpf3d4", &interp<3, 4, float>);
    m.def("interpd1d4", &interp<1, 4, double>);
    m.def("interpd2d4", &interp<2, 4, double>);
    m.def("interpd3d4", &interp<3, 4, double>);
}
