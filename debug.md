# Vlasiator GPU: Debugging Notes and Fix Guide

This document records the two major runtime issues encountered, how we traced them, and the fixes/workarounds. It is intended as a practical guide for future runs (workstations and clusters).

## 1) GPU invalid argument / illegal memory access during block adjustment

- Symptoms
  - compute-sanitizer / CUDA runtime errors around kernels in `spatial_cell_gpu.cpp` (e.g., `resize_vbc_kernel_pre`, `update_velocity_blocks_kernel`).
  - Invalid `cudaMemcpyAsync/cudaMemsetAsync` to host-like pointers for per-thread return buffers.
  - Occasional host segfaults and allocator shutdowns in `split_allocators.h` after long runs.

- Root cause
  - OpenMP was spawning more CPU threads than the code’s per-thread GPU resource arrays could hold. The compile-time cap `MAXCPUTHREADS` was 64 while runs used up to ~80 threads → out-of-bounds writes and memory corruption that later surfaced in various GPU/host operations.

- Diagnostics added (safe for clusters)
  - Pointer attribute checks and guarded GPU memops in `arch/gpu_base.hpp`.
  - Host-side markers and stream sync around each GPU call in `spatial_cell_gpu.cpp`.
  - Device-side bounds/null checks within kernels for list access and container LIDs.
  - Persistent device header mirrors for SplitVector to avoid per-call host/device mismatches.

- Fix implemented
  - Increased per-thread capacity: `#define MAXCPUTHREADS 256` in `arch/gpu_base.hpp`.
  - Added a guard in GPU init to abort if OpenMP threads exceed `MAXCPUTHREADS`.
  - Result: runs complete through physics, deterministic behavior, no more per-thread OOB corruption.

- Operational tip
  - Ensure `OMP_NUM_THREADS <= MAXCPUTHREADS` (or rebuild with a higher cap) on clusters.

## 2) MPI-IO open failure when writing VLSV files on cluster

- Symptoms
  - At first write step, logs show:
    - `[IO-DBG] writeGrid: rank=0 opening '.../bulk.0000000.vlsv' (stripe=0)`
    - `[IO-ERR] writeGrid: open failed for '.../bulk.0000000.vlsv' (rank=0)`
  - Directory exists and is writable from the shell, yet `vlsvWriter.open()` (MPI-IO) fails.
  - Works on desktop; fails on cluster.

- Root cause (environment-specific)
  - The code uses MPI-IO (ROMIO by default) for writing VLSV files. Many clusters disallow or break ROMIO writes to certain filesystems (e.g., home/NFS or some node-local paths). Desktop runs often use local POSIX-friendly filesystems, hence no issue.
  - This repo also explicitly disables OMPIO unless configured otherwise; your run prints: "We detected OpenMPI so we set the cvars value to disable ompio".

- What controls the output location
  - Config keys read in `parameters.cpp`:
    - `io.system_write_file_name` → series name, e.g., `bulk`.
    - `io.system_write_path` → directory (vector per output group). If omitted, defaults to `./`.
  - `iowrite.cpp` constructs the filename from these and calls `vlsvWriter.open()` with MPI-IO.

- Solutions (pick one that matches your cluster policy)
  1) Enable OMPIO instead of ROMIO (often works on NFS/home)
     - Build-time: allow OMPIO override by defining `VLASIATOR_ALLOW_MCA_OMPIO` (uncomment in `Makefile`, or add to your CXX flags).
     - Runtime: set the MCA var before launching:
       - `export OMPI_MCA_io=ompio`
  2) Use a parallel/project filesystem (preferred on many systems)
     - Set in cfg (absolute path):
       - `io.system_write_path = /project/<group>/<user>/vlasiator_run`
     - Ensure the directory exists: `mkdir -p /project/<group>/<user>/vlasiator_run`.
  3) NFS-friendly ROMIO hints (if you must stick with ROMIO)
     - In cfg:
       - `io.system_write_mpiio_hint_key = romio_no_lock`
       - `io.system_write_mpiio_hint_value = 1`
     - Optional:
       - `io.system_write_mpiio_hint_key = romio_ds_write`
       - `io.system_write_mpiio_hint_value = disable`
  4) Node-local scratch test (with OMPIO enabled)
     - In cfg:
       - `io.system_write_path = /tmp/$USER/vlasiator_run`
     - Create before run: `mkdir -p /tmp/$USER/vlasiator_run`.

- Minimal cfg example that works on clusters (adjust paths)

  ```ini
  [io]
  diagnostic_write_interval = 1
  write_initial_state = 0

  system_write_t_interval = 10
  system_write_file_name = bulk
  system_write_path = /project/yourgroup/youruser/vlasiator_run
  system_write_distribution_stride = 0
  system_write_distribution_xline_stride = 10
  system_write_distribution_yline_stride = 10
  system_write_distribution_zline_stride = 1

  # Optional ROMIO hint for NFS
  # system_write_mpiio_hint_key = romio_no_lock
  # system_write_mpiio_hint_value = 1
  ```

- Verification
  - On run start you should see:
    - `[IO-DBG] writeGrid: rank=0 opening '/path/bulk.0000000.vlsv' (stripe=0)`
    - No `[IO-ERR] open failed`.
  - Output file appears in the configured directory and increments with timestep.

## Quick reference

- Where the output filename is formed: `iowrite.cpp` (uses `P::systemWritePath` and `P::systemWriteName`).
- Where cfg is parsed: `parameters.cpp` (`RP::get("io.system_write_*", ...)`).
- Debugging helpers:
  - I/O: [IO-DBG]/[IO-ERR] messages in `iowrite.cpp` indicate the exact failing stage.
  - GPU: host markers and safe guards in `spatial_cell_gpu.cpp`, pointer/debug helpers in `arch/gpu_base.hpp`.

## FAQ

- Why did it work on my desktop but not on the cluster?
  - Different MPI-IO backends and filesystem policies. Desktop likely used local FS with permissive settings; cluster home/NFS can reject ROMIO patterns or require OMPIO or a specific parallel FS.

- Do I need to pre-create `bulk.000000N.vlsv` files?
  - No. The code creates them. You only need to ensure the target directory exists and is writable to MPI-IO.

- Why MAXCPUTHREADS matters?
  - Per-thread GPU arrays are sized to `MAXCPUTHREADS`. Exceeding it with OpenMP leads to out-of-bounds writes and hard-to-trace corruption. We increased it to 256 and added a guard.

---

If you need, we can add a small pre-open check to validate that `P::systemWritePath[...]` exists and is writable (rank 0 creates it, followed by an MPI barrier). But config + correct FS choice is the recommended approach.
