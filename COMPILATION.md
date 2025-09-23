# Compilation Guide

```bash
spack install gcc@10.5.0
spack install cuda@12.3
spack install boost@1.72.0 +program_options %gcc@10.5.0

spack load gcc@10.5.0
spack load cuda@12.3


make ARCH=qichen_fedora_cuda clean
make ARCH=qichen_fedora_cuda -j5
```
