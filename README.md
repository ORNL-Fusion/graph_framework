# Quick Start Guide

**graph_framework**: A graph computation framework that supports auto 
differentiation. It is designed to allow domain scientists to create cross 
platform GPU accelerated code and embed those codes in existing legacy tools. It
is designed for the domain of physics problems where there same physics is 
applied to a large ensemble of independent systems.

This framework enables:
1. Portability to Nvidia, AMD, and Apple GPUs and CPUs.
2. Abstraction of the physics from the compute.
3. Auto Differentiation.
4. Embedding in C, C++, and Fortran codes. 

The compute kernels created have strong scaling to multiple devices

![Strong Scaling](graph_docs/StrongScaling.png)

and the best throughput on both GPUs and CPUs compared to other frameworks like
[MLX](https://ml-explore.github.io/mlx/build/html/index.html) and 
[JAX](https://docs.jax.dev/en/latest/)

![Throughput](graph_docs/Comparison.png)

[![Continuous Integration Test](https://github.com/ORNL-Fusion/graph_framework/actions/workflows/ci_test.yaml/badge.svg?branch=main)](https://github.com/ORNL-Fusion/graph_framework/actions/workflows/ci_test.yaml)

## Documentation
[graph_framework-docs](https://ornl-fusion.github.io/graph_framework-docs) 
Documentation for the graph_framework.

## Obtaining the Code
To get started clone this repository using the command.
```
git clone https://github.com/ORNL-Fusion/graph_framework.git
```

## Compiling the Code
For instructions to build the code consult the 
[build system](https://ornl-fusion.github.io/graph_framework-docs/build_system.html)
documentation. This framework uses a [cmake](https://cmake.org) based build 
system and requires the 
[NetCDF-C](https://www.unidata.ucar.edu/software/netcdf/) library.
