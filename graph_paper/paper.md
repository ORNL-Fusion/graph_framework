---
title: 'graph_framework: A Domain Specific Compiler for Building Physics Applications'
tags:
    - C++
    - Autodifferentation
    - GPU
    - RF Ray Tracing
    - Energenic particles
authors:
    - name: M. Cianciosa
      orcid: 0000-0001-6211-5311
      affiliation: "1"
    - name: D. Batchelor
      orcid: 0009-0000-2669-9292
      affiliation: "2"
    - name: W. Elwasif
      orcid: 0000-0003-0554-1036
      affiliation: "1"
affiliations:
    - name: Oak Ridge National Laboratory
      index: 1
    - name: Diditco, Oak Ridge TN 37831
      index: 2
date: 22 Sepember 2025
bibliography: paper.bib
---

# Summary[^1]

Modern supercomputers are increasingly relying on Graphic Processing Units (GPUs) 
and other accelerators to achieve exa-scale performance at reasonable energy 
usage. The challenge of exploiting these accelerators is the incompatibility 
between different vendors. A scientific code written using CUDA will not operate 
on a AMD gpu. Frameworks that can abstract the physics from the accelerator 
kernel code are needed to exploit the current and future hardware. In world of 
machine learning, several auto differentiation frameworks have been developed 
that have the promise of abstracting the math from the compute hardware. However 
in practice, these framework often lag in supporting non-CUDA platforms. Their 
reliance on python makes them challenging to embed within non python based 
applications. In this paper we present the development of a graph computation 
framework which compiles physics equations to optimized kernel code for the 
central processing unit (CPUs), Apple GPUs, and NVidia GPUs. The utility of this 
framework will be demonstrated for a Radio Frequency (RF) ray tracing problems 
in fusion energy.

[^1]:Notice of Copyright This manuscript has been authored by UT-Battelle, LLC 
under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The 
United States Government retains and the publisher, by accepting the article for 
publication, acknowledges that the United States Government retains a 
non-exclusive, paidup, irrevocable, world-wide license to publish or reproduce 
the published form of this manuscript, or allow others to do so, for United 
States Government purposes. The Department of Energy will provide public access 
to these results of federally sponsored research in accordance with the DOE 
Public Access Plan ([http://energy.gov/downloads/doe-public-access-plan](http://energy.gov/downloads/doe-public-access-plan)).

# Statement of need

Fusion energy is a grand engineering challange to make into a viable power 
source. Beyond the technical challanges towards making it work in the first
place, there is an economic challange that it needs to be addressed. For fusion 
energy to be competative in the market place. Addressing the economic challange 
is avoided though design optimization. However, a barrier to optimization is the 
computational costs associated with exploring the different configurations. 

Low fiedelity models like systems codes, can miss critical physics that enable 
optimized designs. High fidelity models, are too costly to run for multiple 
configurations. GPUs offer tremendous processing power that is largly untapped 
in codes developed by domain scientists. Due to the challanges of exploiting 
GPUs they are largely religated to hero class codes which can use a large 
precentage of Exa-scale machines.

However, there is an intermediate scale of codes which individually can operate
using modest computational requirements but become a challange when generating 
large enembles. These codes are typically CPU only due to the challanges of 
adopting GPUs. As more super computers are deminishing CPU capacity in favor of
GPU support, we are losing the capacity computing necessary to explore large 
ensembles necessary for device optimization.

The goal of the graph_framework is to lower the barrier of entry for adopting 
GPU code. While there are many different solutions to the problem of performance 
portable code, different solutions have different drawbacks or trade offs. With 
that in mind the graph_framework was developed to address the specific 
capabilities of:

- Transparently support multiple CPUs and GPUs including Apple GPUs.
- Use an API that is as simple as writting equations.
- Allow easy embedding in legacy code (Doesn't rely on python).
- Enables automatic differentiation.

With these design goals in mind this framework is limited to the classes of 
problems which the same physics is applied to a large ensemble of particles.
This limitation simplifies the complexity of this framework making future 
extensibility simpler as a need arises for a new problem domain. In this paper 
will discribe the frameworks design and capabilities. Demonstraite applications 
to problems in radio freqiency (RF) heating and particle tracing, and show its 
performance scaling.

# Background

| Framework       | Languauge          | Cuda Support       | Metal Support        | RocM Support       | Auto Differentation |
|:---------------:|:------------------:|:------------------:|:--------------------:|:------------------:|:-------------------:|
| graph_framework | C++, C, Fortran    | Offical            | Offical              | Preliminary        | Yes                 |
|-----------------|--------------------|--------------------|----------------------|--------------------|---------------------|
| Cuda            | C                  | Offical            | None                 | None               | No                  |
| Metal           | Objective C, Swift | None               | Offical              | Depricated         | No                  |
| Kokkos          | C++                | Offical            | None                 | Offical            | No                  |
| OpenACC         | C, C++, Fortran    | Offical            | None                 | None               | No                  |
| OpenMP          | C, C++, Fortran    | Compiler Dependent | None                 | Compiler Dependent | No                  |
| OpenCL          | C                  | Offical            | Depricated           | Offical            | No                  |
| Vulcan          | C                  | Offical            | Unoffical            | Offical            | No                  |
| HIP             | C                  | Offical            | None                 | Offical            | No                  |
| tensorflow      | Python, C++        | Offical            | Unoffical/Incomplete | Unoffical          | Yes                 |
| JAX             | Python             | Offical            | Unoffical/Incomplete | Offical            | Yes                 |
| pytorch         | Python, C++, Java  | Offical            | Offical              | Offical            | Yes                 |
| mlx             | Python, C++, Swift | Offical            | Offical              | Experimental       | Yes                 |
Table: Overview of GPU capable frameworks. \label{frameworks}

Standardized programming languages such as Fortran[@Backus], C[@Ritchie], 
C++[@Stroustrup], have simplified the 
development if cross platform programs. Scientific codes have relied on the 
ability to write source code which can operate on multiple processor 
architectures and operating systems (OSs) with no or little changes given an
appropriate compiler. However, modern super computers rely on graphical 
processing units (GPUs) to achieve exa-scale 
performace[@Hines],[@Yang],[@Schneider] with reasonable energy usage. Unlike 
central processing units (CPUs), the instruction sets of GPUs are proprietary 
information. Additionally, since accelerators typically are hardware 
accessories, an OS requires device drivers which are also proprietary. NVidia
GPUs are best programmed using CUDA[@Cuda] while Apple GPUs use Metal[@Metal] 
and AMD GPUs use HIP[@Hip].

There are many potential solutions to cross performance portable support. Low 
level cross platform frameworks general purpose GPU (GPGPU) programming 
frameworks such as OpenCL[@Munshi] and Vulkan[@Vulkan] requires 
direct vendor support. HIP can support NVidia GPUs by abstracting the driver API
and rewitting kernel code. However these frameworks are the lowest level and
require GPU programming expertize to utilize them effectively that a domain 
scientist may not have. A higher level approch used in 
OpenACC[@Farber] and OpenMP[@OpenMP] use source code antonation to
transform loops and code blocks into GPU kernels. The drawback of this approche
is that source code written for CPUs is results in poor GPU performance. 
Kokkos[@Edwards] is a collection of performance portable array operations for 
for building device adnostic applications.

With the advent of Machine learning, several machine learning frameworks have
been created such as Tensorflow[@Abadi], 
JAX[@Bradbury], PyTorch[@Paszke], and MLX[@Hannun]. These 
frameworks build a graph representation operations that can be 
auto-differentiated and compiled to GPUs. These frameworks are intended to be 
used through a python interface which lower the one barrier to useing but also
introduces new barriers. For instance, it's not straight forward to embed these 
frameworks in non-python codes and non-python API's don't always support all the
features or are as well documented as python API's. Addtitionally performance is
not garrentteed. It is not always straight forward to understand what the 
framework is doing. Additionally cross platform support is often unoffical and
can be incomplete. Table \ref{frameworks} shows an overview of these frameworks.

# Discription

The basic functionality of this framework is to build expression graphs 
representing mathmatical equations. Reduce those graphs to simplier forms. 
Transform those graph to take derivatives. Just-In-Time (JIT) compile them to
available compute device kernels. Then run those kernels in workflow. The code 
is written in using C++23 features. To simplify embedding into legacy codes, 
there are additional language bindings for C and Fortran.

## Graphs

![Mathematical operations are defined as a tree of operations. A `df` method transforms the tree by applying the derivative chain rule to each node. A `reduce` method applies algebraic rules removing nodes from the graph.\label{fig:Tree}](Tree.png){width=60%}

The foundation of this framework is build around a tree data structure that 
enables the symbolic evaluation of mathematical expressions. The `graph` 
namespace contains classes which symbolically represent mathematical operations
and symbols. Each node of the graph is defined as a class dervied from a 
`leaf_node` base class. The `leaf_node` class defines method to `evaluate`, 
`reduce`, `df`, `compile`, and method for introspection. A feature unique to 
this framework is the expression trees can be rendered to \LaTeX  allowing a
domain physicst to understand what the result of reductions and transformations. 
This can also be used to identify future reduction opportunities.

An important distinction of this framework compared to other auto 
differentiation frameworks is there is no distintion between nodes representing
operations and nodes representing values. Sub-classes of `leaf_node` include 
nodes for constants, variables, arithmetic, basic math functions, and 
trigonometry functions. Other nodes encapsulate more complex expressions like 
piecewise constants which depend on the evaluation of an argument. These 
piecewise constants are used implement spline interpolation expressions.

Each node is constructed via factory methods. For common arithmetic operations, 
the framework overloads the $+-*/$ operators to construct expression nodes. The 
factory method checks a `node_cache` to avoid building duplicate sub-graphs.
Identification of duplicate graphs is performed by computing a hash of the 
sub-graph. This hash can be rapidly checked if the same hash already exists in a 
`std::map` container. If the sub-graph already exists, the existing graph is 
returned otherwise a new sub-graph is registered in the `node_cache`.

Each time an expression is built, the `reduce` method is called to simplify the 
graph. For instance, a graph consisting of constant added to a constant will be 
reduced to a single constant by calling the `evaluate` method. Sub-graph 
expressions are combined, factored out, or moved to enable better reductions on 
subsequent passes. As new ways of reducing the graph are implemented, current 
and existing code built using this framework benefit from improved speed. Figure 
\ref{fig:Tree} shows a visualization of the tree data structure for the equation
of a line, the derivative, and the subsequent reductions.

### Building Graphs example

As an example building an expression of line $y=mx+b$ accomplished by creating 
a `variable` node then applying operations on that node.
```
auto x = graph::variable<float> (10, "x");
auto y = 0.5*x + 0.1;
```
In this example, we have created a `variable` with the symbol $x$ containing 
10 elements. Then built the expression tree for $y$. Derivatives are taken using 
the `df` method.
```
auto dydx = y->df(x);
```
Reductions are performed transparently as expressions are created so the 
expression for $\frac{\partial y}{\partial x}=0.5$. As noted before, since this 
framework makes no distinction between the various parts of a graph, derivatives 
and also be taken with respect to sub-expression.
```
auto dydmx = y->df(0.5*x);
```
In this case, the result will be $\frac{\partial y}{\partial 0.5*x}=1.0$

## Workflows

A workflow manager is responsible for compiling device kernels, and running them
in order. One workflow manager is created for each device or thread. The user is 
responsible for creating threads. Each kernel is generted through a work item. A 
work item is defined by kernel inputs, outputs and maps. A map items are used to 
take the results of kernel and update an input buffer. Using out example of line
equation, we can create a workflow to compute $y$ and 
$\frac{\partial y}{\partial x}$.
```
workflow::manager<T> work(0);
work.add_item({
    graph::variable_cast(x)
}, {
    y,
    dydx
}, {}, NULL, "example_kernel", 10);
```
Here we have defined a kernel called "example_kernel". It has one input $x$, two 
outputs $y$ and $\frac{\partial y}{\partial x}$, and no maps. The `NULL` 
argument signifies there is no random state used. The lasy argument needs to 
match the number of elements in the inputs Multiple work items can be created 
and will be executed in order of creation.

Once the work items are defined that can be JIT compiled to a backend device. 
The graph framework supports back ends for generic CPUs, Apple Metal GPUs, 
Nvidia Cuda GPUs, and initial HIP support of AMD GPUs. Each back end supplies 
relevant driver code to build the kernel source, compile the kernel, build 
device data buffers, and handle data synchronization between the device and 
host. All JIT operations are hidden behind a generic `context` interface.

Each context, creates a specific kernel preamble and post-fix to build the 
correct syntax. Memory access is controlled by loading memory once in the 
beginning, and storing the results once at the end. Kernel source code is built 
by recursively traversing the output nodes and calling the `compile` method of each 
`leaf_node`. Each line of code is stored in a unique register variable assuming 
infinite registers. Duplicate code is eliminated by checking if a sub-graph has 
already been traversed. Once the kernel source code is built, the kernel library
is compiled, and a kernel dispatch function is created using a C++ lambda 
function. A workflow can be multiple times.
```
work.compile();
for (size_t i = 0; i < 100; i++) {
    work.run();
}
work.wait();
```
While this API is more explicit compared to the capabilities of JAX, Pytorch, 
TensorFlow, and MLX, it doesn't result in unexpected situations where graphs are
being rebuilt and the user can trust when evaluation is finished. Additionally 
device buffers are only created for kernel inputs and outputs allowing the user
to explicitly control memory useage.

# Use Cases

There are many problems in fusion energy where the same equation needs to be
applied to a large ensemble. This paper will highlight two examples using the 
graphframe work. The first is an RF ray tracing problem to determine plasma 
heating. The second example is for particle pushing.

## RF Ray tracing

![Ray trajectory for $1\times10^{5}$ rays traced in a realistic tokamak geometry.\label{fig:TokamakRays}](TokamakRays.png){width=40%}

![Ray trajectory for $1\times10^{4}$ rays traced in a realistic stellarator geometry using the same dispersion relation and integrator as Figure \ref{fig:TokamakRays}.\label{fig:StellaratorRays}](StellaratorRays.png){width=80%}

Geometric optics is a set of asymptotic approximation methods to solve wave 
equations. The physics of the particular wave determines an algebraic relation 
between $\omega$ and $\vec{k}$ called a dispersion relation, 
$D\left(\omega,\vec{k}\right)=0$. Since the parameter $t$ does not appear 
explicitly in the dispersion relation, the function 
$\omega\left(\vec{k}\left(t\right),\vec{x}\left(t\right)\right)$ is constant 
along the ray trajectory
$$
\frac{\partial\omega}{\partial t}=\frac{\partial\omega}{\partial\vec{x}}\cdot\frac{\partial\vec{x}}{\partial t}+\frac{\partial\omega}{\partial\vec{k}}\cdot\frac{\partial\vec{k}}{\partial t}\equiv 0
$$
by virtue of the ray equations. Since the dispersion relation is satisfied all 
along the ray trajectory, the derivatives needed for the ray equations can be 
obtained by implicit differentiation
$$
\frac{\partial D}{\partial\vec{x}}=\frac{\partial D}{\partial\omega}\frac{\partial\omega}{\partial\vec{x}}\Rightarrow\frac{\partial\omega}{\partial\vec{x}}=-\frac{\frac{\partial D}{\partial\vec{x}}}{\frac{\partial D}{\partial\omega}}
$$
$$
\frac{\partial D}{\partial\vec{k}}=\frac{\partial D}{\partial\omega}\frac{\partial\omega}{\partial\vec{k}}\Rightarrow\frac{\partial\omega}{\partial\vec{k}}=-\frac{\frac{\partial D}{\partial\vec{k}}}{\frac{\partial D}{\partial\omega}}
$$
These equations are integrated to trace the ray.

A ray tracing problem is build by implementing expressions for the plasma 
equilibrium. From the plasma equilibrium a dispersion relation is constructed.
Equations of motion are defined using the auto differentiation. Expressions for 
ray update are built for in integrator. These expressions are JIT compiled into 
a single kernel call with inputs for $\vec{x}$, $\vec{k}$, $t$, and $\omega$ 
with outputs for the dispersion residual, and step updates for $\vec{x}$, 
$\vec{k}$ and $t$. Figure \ref{fig:TokamakRays} shows $1\times10^{5}$ O-Mode 
rays traced in a realistic tokamak geometry. Figure \ref{fig:StellaratorRays}
shows $1\times10^{4}$ ray trajectories for a stellarator equilibrium using the 
same dispersion, and integrator method.

## Particle Pushing

![Particle trajectories in a realistic tokamak geometry.\label{fig:TokamakParticles}](ParticleTraces.png){width=80%}

In order to achieve good statistics for the evolution of particle distributions, 
it's necessary to push large numbers of particles. Exploiting GPU resources is 
necessary to achieve the large number of particles at reasonable run times to 
enable self consistent fields. An example is the runaway electron problem.

During a disruption in a tokamak, electric fields can drive electron beams up to 
relativistic speeds. These high energy particles can lose confinement and damage 
first wall components. The Boris leap-frog algorithm can integrate particles 
while conserving energy and momentum[@Tamburini]. The algorithm updates
particle position $\vec{x}$, momentum $\vec{u}$, and relativistic $\gamma$.
Figure \ref{fig:TokamakParticles} shows $100$ out of $1\times10^{5}$ particles 
trajectories in a realistic tokamak geometery.

# Code Performance

![Left: Strong scaling wall time for 1E6 Rays traced in a realistic tokamak equilibrium. Right: Strong scaling speedup normalized to the wall time for a single device or core. The dashed diagonal line references the best possible scaling. The M2 Max has 8 fast performance cores and 4 slower energy efficiency cores resulting drop off in improvement beyond 8 cores.\label{fig:benchmark}](StrongScaling.png){width=100%}

To benchmark code performance we traced $1\times 10^{6}$ rays using the cold 
plasma dispersion relation in a realistic tokamak equilibrium. Figure 
\ref{fig:benchmark} shows the strong scaling of wall time as the number of 
GPU and CPU devices are increased. Figure \ref{fig:benchmark} shows the 
strong scaling speed up
$$
SpeedUp = \frac{time\left(1\right)}{time\left(n\right)}
$$
Benchmarking was prepared on two different setups.

The first set up as a Mac Studio with an Apple M2 Max chip. The M2 chip contains 
a 12 core CPU where 8 cores are faster performance codes and the remaining 4 are 
slower efficiency cores. The M2 Max also contains a single 38-core GPU which 
only support single precision operations. The second setup is a server with 4 
Nvidia A100 GPUs. Benchmarking measures the time to trace $1\times10^{6}$ rays 
but does not include the setup and JIT times.

Figure \ref{fig:benchmark} shows the advantage even a single GPU has over 
CPU execution. In single precision, the M2's GPU is almost $100\times$ faster a 
single CPU core while the a single A100 has a nearly $800\times$ advantage. An 
interesting thing to note is the M2 Max CPU show no advantage between single and
double precision execution.

For large problem sizes the framework is expected to show good scaling with 
number of devices as the problems we are applying are embarrassingly parallel in 
nature. Figure \ref{fig:benchmark} shows the strong scaling speed up with the 
number of devices. The framework shows good strong scaling as the problem is 
split among more devices. The architecture of the M2 Chip contains 8 fast 
performance cores and 4 slower energy efficiency cores. This produces a 
noticeable knee in the scaling after 8 core are used. Overall, the framework 
demonstrates good scaling across CPU and GPU devices.

## Comparison to other frameworks

To beckmark against other frameworks we will look at the simple case of a gyro 
motion in a uniform magnetic field.

# Acknowledgements
The authors would like to thank Dr. Yashika Ghai, Dr. Rhea Barnett, and Dr. 
David Green for their valuable insights when setting up test cases for the 
RF-Ray Tracing.
