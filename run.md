# Compilation Guide

## Setup

```bash
spack install gcc@10.5.0
spack install cuda@12.3

spack install boost@1.72.0 +program_options %gcc@10.5.0
or build boost from source 

spack install openmpi@4.1.1 +cuda %gcc@10.5.0

spack load gcc@10.5.0
spack load cuda@12.3


make ARCH=qichen_fedora_cuda clean
make ARCH=qichen_fedora_cuda -j5
```

## Monitoring

```bash
nvdia-smi # GPU
htop # CPU
```


## Notice

Add `-lnvToolsExt` for nvtx profiling in `Makefiles`. Currently there are errors when compiling with nvtx: `libnvToolsExt.so` not found.

## Example output

### GPU

```bash
time ../../../vlasiator --run_config Magnetosphere_small.cfg
(Grid) rank 0 is noderank 0 of 1
Done setting all 63 instances of device mesh wrapper handler!
(MAIN): Completed grid initialization.
(MAIN): Starting main simulation loop.
(MAIN): Completed requested simulation. Exiting.
../../../vlasiator --run_config Magnetosphere_small.cfg  936.58s user 73.10s system 387% cpu 4:20.71 total
```

GPU usage is around 240 MB.

### CPU

```bash
time ../../../vlasiator --run_config Magnetosphere_small.cfg
../../../vlasiator --run_config Magnetosphere_small.cfg  253.98s user 1.06s system 485% cpu 52.513 total
```

Add

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/boost/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/phiprof/lib
```

to solve

```shell
../../../vlasiator --run_config Magnetosphere_small.cfg
../../../vlasiator: /home/qichen/Dev/spack/opt/spack/linux-skylake/gcc-10.5.0-57oyfp25wcrxtvem46bfpifk6j3e4n34/lib64/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /lib64/libboost_program_options.so.1.72.0)
../../../vlasiator: /home/qichen/Dev/spack/opt/spack/linux-skylake/gcc-10.5.0-57oyfp25wcrxtvem46bfpifk6j3e4n34/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib64/libboost_program_options.so.1.72.0)

```

Uncomment

```
COMPFLAGS += -DVLASIATOR_ALLOW_MCA_OMPIO
```

in `Makefile` and

```
export OMPI_MCA_io=ompio
```

in console
