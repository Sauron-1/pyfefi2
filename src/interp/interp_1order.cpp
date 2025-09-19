#include <interp/interp.hpp>
#include <interp/interp_ext.hpp>

void init_interp_ext_1order(py::module_& m) {
    m.def("interpf1d1", &interp<1, 1, float>);
    m.def("interpf2d1", &interp<2, 1, float>);
    m.def("interpf3d1", &interp<3, 1, float>);
    m.def("interpd1d1", &interp<1, 1, double>);
    m.def("interpd2d1", &interp<2, 1, double>);
    m.def("interpd3d1", &interp<3, 1, double>);
}
