# Notes on “warp divergence” markers in `construct_columns_kernel`

Context: `vlasiator/vlasovsolver/gpu_acc_sort_blocks.cpp` (kernel around lines ~257–433).

This note reviews the branches marked as “[Warp Divergence]” in `construct_columns_kernel` and summarizes what is
actually divergent (lane-dependent control-flow) vs. what is warp-uniform or intentional serialization. It also lists
practical mitigation options.

## Key context: the kernel is launched as a single block

`construct_columns_kernel` is launched as:

- `construct_columns_kernel<<<1, GPUTHREADS, 0, stream>>>(...)`

So the kernel is already **single-block**, and large parts of the logic are effectively **serial-within-a-block** (using
shared state and a “leader” thread). That makes many “divergence” markers either:

- warp-uniform (all lanes take the same path), or
- leader-thread-only work (lane 0 performs bookkeeping while others wait at `__syncthreads()`).

## Branches that are *not* harmful warp divergence (warp-uniform)

### `switch (dimension)` used to set `DX`

`dimension` is a kernel argument, so it is uniform across the whole block/warp. The `switch` does **not** create
lane-level divergence.

**Possible mitigation (minor)**:

- Specialize by dimension (three entry points or a templated kernel) so the compiler can constant-fold the index and
  potentially reduce instruction count. This is mostly about codegen and hoisting, not divergence.

### “Only one column in columnset?” branch

The predicate depends on `i`, `blocks_in_columnset`, `DX`, and values from `blocksID_mapped_sorted`, but **not** on `ti`.
So all lanes take the same `if/else` path for that iteration. This is warp-uniform control flow.

### `minstep` / ballot-derived decisions

Branches that depend on `minstep` computed from a warp ballot are also warp-uniform: every lane sees the same
`ballot_result` and thus the same `minstep`.

## Branches that are “divergent” only because of intentional serialization

### All `if (ti==0)` leader-thread blocks

Many markers are on branches guarded by `ti==0` (or combined with `ti==0`), e.g. initialization of shared variables,
updating the shared index `i`, and appending offsets/lengths.

This is not accidental divergence: the algorithm maintains and mutates shared state (`i`, `blocks_in_columnset`) and
appends to dynamic arrays, so it is inherently serial in its current form.

Note: the `SplitVector::device_push_back` implementation uses atomics and an overflow branch internally, even though in
this kernel it is invoked from a single lane. The dominant concern is **serial overhead and atomics**, not warp divergence.

## Branches that *do* create real lane-level divergence

### Tail/bounds check inside the warp-sized scan

In the inner scan over a chunk:

- `if (ci + ti < blocks_in_columnset) { ... }`

This diverges only on the last partial chunk (when the remaining elements are fewer than the block’s thread count).
This is the primary true lane-dependent divergence in the kernel.

**Possible mitigation (small / may already be optimized by compiler)**:

- Replace the control-flow branch with predication: compute an `in_range` boolean and fold it into `notInColumn`.
  This can reduce control-flow divergence at the expense of a bit more arithmetic/loads; on many compilers this is already
  effectively predicated.

## Guard branches that currently don’t matter for divergence

### `if (gpuBlocks != 1) return;`

Given the current launch (`<<<1, ...>>>`), this condition is false and has no runtime impact.

**Possible mitigation (cleanup)**:

- Make this a host-side assertion/guard and remove the device-side branch.

## Meaningful mitigation requires restructuring, not micro-tweaks

If the goal is to reduce control-flow complexity and “waiting on lane 0” (and potentially to enable more GPU
parallelism), the biggest lever is to replace the serial `while (i < nBlocks)` walk with a data-parallel construction:

- Build a per-element “break flag” array \(b[i]\) in parallel that indicates whether element `i` begins a new column (or a
  new set), based on comparing `(column_id, dimension_id)` at `i` vs `i-1`.
- Do a parallel prefix-scan of \(b\) to produce a column index per element.
- Scatter column start offsets and compute lengths (e.g., by writing start indices then differencing, or by writing end
  indices similarly).
- Allocate/resize output arrays up-front (two-pass: count then write) to avoid dynamic `device_push_back`.

This approach eliminates most leader-thread-only regions and reduces branch structure substantially, but it is a larger
algorithmic change and likely requires extra temporary storage and/or scan primitives.

## Bottom line

- Most “[Warp Divergence]” tags in `construct_columns_kernel` correspond to **warp-uniform branches** or **intentional
  leader-thread serialization**, not harmful lane-level divergence.
- The main true divergence is the **bounds/tail predicate** in the warp-sized scan, and it is typically a small effect.
- The only way to materially reduce the control-flow/idle-lane behavior is to **restructure the algorithm** to avoid the
  shared index walk and dynamic `device_push_back` (two-pass + scan/compact style).

