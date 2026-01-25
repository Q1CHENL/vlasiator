## Report: review of `lid_shared` “use shared” optimization in `gpu_acc_map.cpp`

**Scope**: Validate the correctness and likely performance impact of caching `gpu_LIDlist[inputOffset + b]` in shared memory inside `reorder_blocks_by_dimension_kernel` (`vlasiator/vlasovsolver/gpu_acc_map.cpp`, around lines 70–150).

### Summary

- **Correctness**: The `__shared__ vmesh::LocalID lid_shared;` change is **correct** as written (no races, no invalid barrier usage, no per-thread semantic change).
- **Safety assumptions**: The code relies on the existing assumption that all threads in the CUDA block execute the same loop trip count for `b` (true here) and reach the `__syncthreads()` (true here).
- **Performance**: The optimization **can reduce redundant loads** of `gpu_LIDlist[inputOffset + b]` but introduces a synchronization point. With `VECL <= 16` typical in Vlasiator, blocks are usually a single warp and the sync cost is lower—but it is still a barrier. Whether it wins depends on compiler hoisting, cache behavior, and how expensive the downstream `gpu_blockData[...]` loads are relative to that saved load.

### Relevant code (current)

In `reorder_blocks_by_dimension_kernel`:

- Declaration:
  - `__shared__ vmesh::LocalID lid_shared;`
- Producer:
  - `if (ti==0) lid_shared = gpu_LIDlist[inputOffset + b];`
- Barrier:
  - `__syncthreads();`
- Consumer:
  - `gpu_blockData[lid_shared * WID3 + sourceindex]`

### Correctness analysis

#### Data-race / memory ordering

- Only one thread writes `lid_shared` per iteration (`ti==0`).
- All threads read `lid_shared` after an unconditional `__syncthreads()`.
- `__syncthreads()` provides the required ordering/visibility for shared memory, so no thread can observe an “old” value from a previous `b`.

**Conclusion**: No race; ordering is correct.

#### Barrier correctness / divergence

- The barrier is inside the `for (uint b = 0; b < columnLength; b++)` loop.
- `columnLength` is computed once per block and is independent of `ti`:
  - `uint columnLength = columnData->columnNumBlocks[iColumn];`
- Therefore, all threads execute the loop with identical bounds and reach `__syncthreads()` the same number of times.

**Conclusion**: No divergent barrier usage.

#### Type/size correctness

- `vmesh::LocalID` is `uint32_t` (`vlasiator/definitions.h`), so `lid_shared` is a 32-bit scalar in shared memory.
- The use `lid_shared * WID3 + sourceindex` remains in-range/overflow-safe iff the original `gpu_LIDlist[inputOffset + b] * WID3 + sourceindex` was safe (this optimization does not change the arithmetic domain).

**Conclusion**: Type/size is consistent with the original code.

### Performance considerations (why this may or may not help)

#### What is saved

Without caching, each thread would need to read:

- `gpu_LIDlist[inputOffset + b]` (same value across all threads) in order to compute the base address for `gpu_blockData[...]`.

With caching, the global load becomes **one per block per `b`** rather than one per thread per `b`.

#### What is added

- A `__syncthreads()` per `b`.
- A shared-memory load of `lid_shared` at the consumer point (typically cheap).

#### Notes specific to this kernel and Vlasiator constants

- `WID` is 4 and `WID3` is 64 (`vlasiator/common.h`).
- The kernel is launched with block size `VECL`:
  - `reorder_blocks_by_dimension_kernel<<<host_totalColumns, VECL, 0, stream>>>(...)`
- `VECL` is commonly 4/8/16 in Vlasiator (`vlasiator/vlasovsolver/vec.h`), meaning blocks are typically **1 warp** (or less).
  - On NVIDIA GPUs, `__syncthreads()` within a single warp still behaves correctly; its cost is typically smaller than multi-warp barriers, but it is not “free”.

**Practical implication**: This optimization is most likely to help when:

- `columnLength` is large enough that redundant LID loads matter, and/or
- `gpu_LIDlist` resides in memory that isn’t consistently cached, and/or
- the compiler cannot safely hoist the `gpu_LIDlist[...]` load out of the inner loops.

It is less likely to matter when the overall kernel is dominated by:

- the `gpu_blockData[...]` global memory traffic (which is much larger than the single LID load), or
- address arithmetic/transpose overhead, or
- any cache/TLB/page-fault behavior from unified memory backing `blockContainer->getData()`.

### Potential refinements (optional)

These are **not required for correctness**, but may be considered if tuning:

- **Register per-thread caching instead of shared**: Load `lid = gpu_LIDlist[inputOffset + b]` into a register in each thread outside the `k/j` loops (still per-thread global loads, but eliminates the barrier). This can win if the barrier is more expensive than the saved global reads and the LID load is cache-hot anyway.
- **Warp-level broadcast**: If you want to keep single-load semantics while avoiding a full-block barrier, you can load in lane 0 and broadcast with `__shfl_sync(...)` (warp-only). This is only valid if the block is guaranteed to be a single warp (i.e., `VECL <= warpSize` and blockDim is 1D). The current code does not explicitly assert this.
- **Hoisting check**: Confirm (via compiler SASS/PTX inspection or profiler) whether the original `gpu_LIDlist[inputOffset + b]` load was already hoisted out of the deepest loops by the compiler. If it was, the shared optimization may not reduce loads.

### Conclusion

The `lid_shared` shared-memory optimization is **legitimate and correct**. It implements a standard “single-load then broadcast” pattern using shared memory and a barrier, and it preserves the original semantics of indexing `gpu_blockData` by `gpu_LIDlist[inputOffset + b]`. Whether it improves runtime is workload- and GPU-dependent, but it is not a correctness risk in its current form.

