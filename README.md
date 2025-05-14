# graph_framework

Graph computation framework that supports auto differentiation.

# Dependencies
This code makes use of the [NetCDF `C` library](https://www.unidata.ucar.edu/software/netcdf/).

On Macs, this requires the latest install of XCode 16.3.

# Compiling
To compile the code, first clone this repository.

A `graph_framework` directory will be created. In this directory create 
a `build` directory and navigate to it.

```
cd graph_framework
mkdir build
cd build
```

It's recommended that you use the `ccmake` to configure the build system. 
From inside the build directory, run the `ccmake` command

```
ccmake ../
```

Initally, there will be no options. Press the `c` key to configure. 
During this step, the LLVM software repo is cloned and configured. 
This may take a while. Once configured, there will be several options.

- On Linux systems, you may optionally use CUDA by toggling the `USE_CUDA` 
option.
- On Mac systems, you may optionally use Metal by toggling the `USE_METAL`
option. 

__NOTE__: On Macs using the default system compiler, you will need to change 
the `CMAKE_CXX_COMPILER` to `clang++`. To do this press the `t` to toggle
to the advanced options.

Once all the options are configured press the 'c' key again and a 
generate 'g' option will be available. Pressing the 'g' key will finish
generating the make file and close out `ccmake`.

One the makefile is created. The code can be built using `make` or 
optionally built in parallel using `make -j'_[number of parallel intances]_.

__NOTE__: To many parallel instances can cause the LLVM build system to
hang so it is recommended to limit these to less than 20 depending on
the available memory.

Unit tests can be run by the `make test` command. An example ray tracing
case can be found in `efit_example.sh` of the `graph_driver` directory.
