//------------------------------------------------------------------------------
///  @file xplayground.cpp
///  @brief A playground area for testing example programs.
//------------------------------------------------------------------------------
#include "../graph_framework/jit.hpp"

#include <numbers>

#include "../graph_framework/equilibrium.hpp"

//------------------------------------------------------------------------------
///  @brief Main program of the playground.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;

//  Insert code here. No code should be commited to this file beyond this
//  template.
    auto r = graph::variable<double> (1000, "r");
    auto phi = graph::variable<double> (1000, "\\phi");
    auto z = graph::variable<double> (1000, "z");

    auto eq = equilibrium::make_efit<double> (EFIT_FILE);

    auto bvec = eq->get_magnetic_field(r*graph::cos(phi),
                                       r*graph::sin(phi),
                                       z);

    bvec->get_x()->to_latex();
    std::cout << "\\\\" << std::endl;
    bvec->get_y()->to_latex();
    std::cout << "\\\\" << std::endl;
    bvec->get_z()->to_latex();
    std::cout << "\\\\" << std::endl;
    std::cout << std::endl << std::endl << std::endl;

    r->set(static_cast<double> (1.7));
    z->set(static_cast<double> (0.0));
    std::vector<double> temp(1000);
    for (size_t i = 0; i < 1000; i++) {
        temp[i] = 2.0*std::numbers::pi_v<double>*i/999.0;
    }
    phi->set(temp);

    workflow::manager<double> work(0);
    work.add_item({
        graph::variable_cast(r),
        graph::variable_cast(phi),
        graph::variable_cast(z)
    }, {
        bvec->get_x(),
        bvec->get_y(),
        bvec->get_z()
    }, {}, "bvec_kernel");
    work.compile();
    work.run();
    for (size_t i = 0; i < 1000; i++) {
        work.print(i, {r, phi, z, bvec->get_x(), bvec->get_y(), bvec->get_z()});
    }

    END_GPU
}
