//------------------------------------------------------------------------------
///  @file xplayground.cpp
///  @brief A playground area for testing example programs.
//------------------------------------------------------------------------------
#include "../graph_framework/graph_framework.hpp"

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

//  Insert code here. No code should be committed to this file beyond this
//  template.
    const size_t num_mesh = 10;
    const size_t num_particles = 100;

    auto xmesh = graph::variable<double> (num_mesh, "x_mesh");
    auto ymesh = graph::variable<double> (num_mesh, "y_mesh");

    const double xmin = -3;
    const double xmax = 3;
    const double dx = (xmax - xmin)/(num_mesh - 1);

    for (size_t i = 0; i < num_mesh; i++) {
        graph::variable_cast(xmesh)->data()[i] = dx*i + xmin;
        graph::variable_cast(ymesh)->data()[i] = std::sin(std::exp(graph::variable_cast(xmesh)->data()[i]));
    }

    auto xp = graph::variable<double> (num_particles, "xp");
    const double dxp = (xmax - xmin)/(num_particles - 1);
    for (size_t i = 0; i < num_particles; i++) {
        graph::variable_cast(xp)->data()[i] = dxp*i + xmin;
    }

    auto x = graph::index_1D(xmesh, xp, dx, xmin) - xp;
    auto xnorm1 = 1.5 + (x - dx)/dx;
    auto xnorm2 = x/dx;
    auto xnorm3 = 1.5 - (x + dx)/dx;

    auto w0 = 0.5*xnorm1*xnorm1;
    auto w1 = 0.75 - xnorm2*xnorm2;
    auto w2 = 0.5*xnorm3*xnorm3;

    (1.5 - ((x + dx)/dx))->to_latex();
    std::cout << std::endl;

    auto weigth = w0 + w1 + w2;
    
    workflow::manager<double> work(0);
    work.add_item({
        graph::variable_cast(xmesh),
        graph::variable_cast(xp)
    }, {
        x,
        weigth,
        w0,
        w1,
        w2
    }, {}, NULL, "Mesh_Interpolation", num_particles);
    work.compile();

    output::result_file file("/Users/m4c/Projects/graph_framework/build/mesh.nc", num_particles);
    output::data_set<double> dataset(file);
    dataset.create_variable(file, "x", x, work.get_context());
    dataset.create_variable(file, "xp", xp, work.get_context());
    dataset.create_variable(file, "weigth", weigth, work.get_context());
    dataset.create_variable(file, "w0", w0, work.get_context());
    dataset.create_variable(file, "w1", w1, work.get_context());
    dataset.create_variable(file, "w2", w2, work.get_context());

    file.end_define_mode();

    work.run();
    work.wait();

    dataset.write(file);

    END_GPU
}
