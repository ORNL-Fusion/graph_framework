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

GPUs offer a offer tremendous processing power that is largly untapped codes 
developed by domain scientists. The goal of the graph_framework is to lower the 
barrier of entry for adopting GPU code. While there are many different solutions
to the problem of performance portable code, different solutions have different
drawbacks or trade offs. With that in mind the graph_framework was developed to
address the specific capabilities of:

- Transparently support multiple CPUs and GPUs including Apple GPUs.
- Use an API that is as simple as writting equations.
- Allow easy embedding in legacy code (Doesn't rely on python).
- Enables automatic differentiation.

With these design goals in mind this framework is limited to the classes of 
problems which the same physics is applied to a large ensemble of particles.
This limitation simplifies the complexity of this framework making future 
extensibility simpler as a need arises for a new problem domain.

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
architectures and operating systems (OSs) with no or littel changes given an
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
