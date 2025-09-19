#include <interp/interp.hpp>
#include <interp/interp_ext.hpp>

void init_interp_ext_2order(py::module_& m) {
    m.def("interpf1d2", &interp<1, 2, float>);
    m.def("interpf2d2", &interp<2, 2, float>);
    m.def("interpf3d2", &interp<3, 2, float>);
    m.def("interpd1d2", &interp<1, 2, double>);
    m.def("interpd2d2", &interp<2, 2, double>);
    m.def("interpd3d2", &interp<3, 2, double>);
}
