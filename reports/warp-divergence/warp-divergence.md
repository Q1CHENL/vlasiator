
# Warp divergence deep dive (Vlasiator + submodules)

This document analyzes all places marked with `[Warp Divergence]` (and variants) in the codebase, focusing on:

- **Whether the divergence is avoidable** (with practical refactors), **inevitable** (algorithmic / correctness), or **benign** (leader-lane / tail handling / structural).
- **Why** the profiler flags it (warp-non-uniform control flow, variable loop trip counts, atomic outcomes, etc.).
- **What optimization levers exist**, ranked by expected payoff.

> Notes on interpretation:
>
> - A profiler flags “warp divergence” for *any* non-uniform branch. Many marked sites are **intentional** (e.g., `tid==0` leader work) and have negligible impact.
> - The expensive cases are typically those where divergence implies **extra work**: **different loop trip counts per lane**, repeated probing, retries after atomics, or divergent early exits inside a hot loop.
> - Some marks appear in code that is launched with 1 thread / 1 block (“serial kernels”): divergence there is **not performance relevant**.

---

## Classification rules (used throughout)

- **Benign / structural**
  - `if (tid < d)` in parallel reductions/scan phases.
  - `if (tid == 0)` / `if (w_tid == 0)` one-time init / writing a scalar.
  - Tail guards: `if (tid < N)` only diverges for the last partial warp(s).
  - Kernel-constant branching: `switch(dimension)` where `dimension` is uniform across the grid/warp.

- **Inevitable / algorithmic**
  - Open addressing hash probing loops (trip count depends on collisions).
  - Branching on **atomic CAS results** (`old == EMPTYBUCKET`, `old == key`).
  - Stream compaction / select-by-predicate: `if (active)` is fundamental.
  - Slope limiters / monotonicity “collapse” logic in reconstruction schemes.
  - Sparsity-driven early-outs that skip expensive reconstruction (beneficial divergence).

- **Potentially optimizable**
  - Divergence that causes **variable loop bounds** per lane in a hot loop.
  - Divergence that can be eliminated by **preconditions** (e.g., “indices are always valid”).
  - Divergence that stems from device-side dynamic container growth / failure paths (`device_push_back`, `device_resize`) under capacity pressure.

---

## `velocity_mesh_gpu.h` (VelocityMesh warp accessors + ID conversions)

### `VelocityMesh::getGlobalID(i,j,k)` bounds checks
Marked at `velocity_mesh_gpu.h:504`:

- **What it is**: three early-return checks:
  - `if (i >= gridLength[0]) return invalidGlobalID();` etc.
- **Classification**: **potentially optimizable** / sometimes **inevitable**
  - If callers already ensure `i,j,k` are within bounds, this can be moved to a debug-only check and removed from hot kernels.
  - If callers may request out-of-range neighbors (halo logic), the guard is necessary.
- **Why profiler flags it**: if some lanes compute out-of-range indices while others don’t.
- **Optimization levers**:
  - Split workloads into **interior** vs **boundary** kernels (interior runs without bounds checks).
  - In halo code, prefer **masking** work with a uniform predicate when possible (e.g., precompute “valid neighbor offsets” for the whole warp), but this is often geometry-dependent.

### `VelocityMesh::getIndices*()` invalidGlobalID branch
Marked at `velocity_mesh_gpu.h:521,545,561,578`:

- **What it is**: `if (globalID >= invalidGlobalID()) { i=j=k=invalidBlockIndex(); } else { compute via div/mod }`
- **Classification**: usually **benign** or **avoidable**
  - If `globalID` is known valid, remove/avoid the branch by not calling these with invalid IDs.
  - If invalid IDs are used as sentinels in vectors/maps, the guard is correct.
- **Why profiler flags it**: mixed valid/invalid `globalID` values within a warp.
- **Optimization levers**:
  - Separate “valid-only” paths.
  - If invalid values are rare, the divergence cost is usually small compared to the div/mod cost and global memory.

### `VelocityMesh::warpPush_back(globalID, b_tid)`
Marked at `velocity_mesh_gpu.h:1004,1016,1023`:

Key branches:

- Capacity/invalid guard:
  - `if (mySize >= max_velocity_blocks) return false;`
  - `if (globalID == invalidGlobalID()) return false;`
  - **Classification**: often **warp-uniform** in practice (all lanes insert same `globalID`), so profiler may still flag but impact is small.

- “Active lanes” guard:
  - `if (b_tid < GPUTHREADS) { ... }`
  - **Classification**: **benign** (uses a warp subset when the block has more threads than a warp).

- Leader lane action:
  - `if (inserted == true && b_tid==0) { device_push_back; ltg_size++; }`
  - **Classification**: **benign/intentional**.

**Real performance risks here are usually not divergence**, but:
- Hash insertion cost (`warpInsert_V`) and memory traffic,
- contention / serialization due to atomics inside hashinator,
- and any container capacity pressure leading to failure paths.

**Optimization levers**:
- Ensure `max_velocity_blocks` sizing is sufficient for expected halo growth.
- Reduce frequency of insertions (batch operations / pre-reserve).
- Prefer “bulk add” paths where you can precompute required blocks and insert once per block/warp.

---

## Hashinator (`submodules/hashinator/include/hashinator/*.h`) and SplitVector (`splitvector/*`)

The largest *meaningful* divergence is in hash probing + atomics. The key point:

> For open addressing hash tables, divergence is tightly coupled to **extra work** (probe steps, retries, and contention). The best way to reduce divergence is to **reduce collisions**.

### `submodules/hashinator/include/common.h`: `nextOverflow()`
Marked at `include/common.h:56`:

- **What it is**: `return (remainder==0) ? currentOverflow : currentOverflow + (virtualWarp - remainder);`
- **Classification**: **benign**
  - Often compiles to predicated instructions. Also typically used in leader/atomic-update contexts, not in per-lane hot loops.
- **Optimization**: not worth pursuing unless it appears in a top-hot inner loop (unlikely).

### `splitvector/split_allocators.h`: `split_unified_allocator::construct()`
Marked at `split_allocators.h:125`:

- **What it is**: placement-new `::new(p) U(args...)`.
- **Classification**: **noise / depends on U**
  - No branch here. Any divergence is in `U`’s constructor, not the allocator wrapper.

### `splitvector/splitvec.h`

#### `_rangeCheck()` / `.at()` path
Marked at `splitvec.h:108`:

- **Classification**: **debug-only / avoid in hot device code**
- **Optimization**:
  - Do not use `.at()` in hot device kernels.
  - Prefer unchecked indexing (`operator[]` / raw pointer arithmetic) when correctness is ensured.

#### `device_resize(newSize, construct=true)`
Marked at `splitvec.h:912,917`:

- `if (newSize > capacity()) assert(0);` is a catastrophic branch.
- The per-element construction loop `for (i=size(); i<newSize; ++i) construct(...)` can be expensive.
- **Classification**:
  - Divergence itself is usually **benign** (warp-uniform capacity checks).
  - The **device-side resize + construct** is the real cost center.
- **Optimization levers**:
  - Avoid resizing from within hot kernels; resize on host or via dedicated “setup” kernels.
  - If safe, call with `construct=false` and explicitly initialize only what you need.

#### `device_push_back()` overflow check
Marked at `splitvec.h:1090,1113`:

- `old = atomicAdd(_size, 1)` then `if (old >= capacity()-1) { atomicSub; return false; }`
- **Classification**: **inevitable** when you allow capacity pressure
  - Whether any lane “fails” depends on racing allocation and current size.
- **Highest-payoff optimization**:
  - Avoid per-lane push_back. Use **block-level reservation**:
    - One atomic per block to reserve a contiguous output range,
    - then each lane writes deterministically into its slot (no overflow branching unless the whole block can’t reserve).

### `splitvector/split_tools.h`: scan + compaction primitives

#### `split_prescan`: `if (tid < d)` and `if (tid == 0)`
Marked at `split_tools.h:166,179,190`:

- **Classification**: **benign structural**
  - This is the standard Blelloch scan shape. Divergence here is expected and usually not the dominating cost compared to syncs and shared memory.
- **Optimization**:
  - Use CUB/hipCUB primitives (`BlockScan`, `DeviceScan`) if scan becomes a top kernel.

#### `split_compact_raw`: prefix offsets by a single lane
Marked at `split_tools.h:485`:

- `if (w_tid == 0 && wid % warps_in_block == 0)` does a serial loop to build warp offsets.
- **Classification**: **benign/intentional**
  - It is leader-lane work; real cost is small compared to global memory traffic of compaction.
- **Optimization**:
  - Consider using CUB select if you want best-in-class compaction.

#### `block_compact` / `loop_compact*`: predicate-driven select
Marked at `split_tools.h:652,661,678` and in loop kernels (`790+` etc.):

- `active = (tid < inputSize) ? rule(...) : false;`
- `if (active) output[...] = input[...]`
- **Classification**: **inevitable**
  - Compaction/selection is fundamentally data-dependent.
- **Optimization levers**:
  - Pad inputs to warp multiples (reduce tail divergence).
  - Reduce sync frequency; reduce shared memory bank conflicts; or switch to CUB primitives.

---

## Hashinator warp-synchronous hashing

### `hashinator/kernels_NVIDIA.h`: bulk insert kernels
Representative marked patterns (multiple sites):

- `if (proper_w_tid == 0 && blockWid == 0) { init shared arrays }`
  - **Classification**: **benign** (one-time per block init).

- Probe loop with voting:
  - `while (mask && !vWarpDone) { winner = ffs(mask); if (w_tid==winner) atomicCAS/atomicExch ... }`
  - **Classification**: **inevitable/algorithmic**
  - **Why it diverges**:
    - Winner lane is unique (intentional).
    - Trip count depends on collisions and which buckets are empty.
    - CAS outcomes cause path differences (`old == EMPTYBUCKET` vs `old == key` vs other).

- Fill/overflow updates reduced to 1 lane:
  - `if (blockWid == 0) { ... if (proper_w_tid == 0) atomicMax/atomicAdd }`
  - **Classification**: **benign** (reduces atomics; good).

**Highest-payoff optimization levers (hashinator overall)**:

1. **Reduce collisions / shorten probe chains**
   - Use a lower **target load factor** (grow earlier).
   - Ensure the hash function is good for your key patterns (block IDs often have structure).
   - Keep `currentMaxBucketOverflow` bounded by preventing pathological overflows.

2. **Tune virtual warp width (`elementsPerWarp`)**
   - Wider virtual warp: more parallel probing but more contention; narrower: less contention but more steps.
   - Best choice depends on occupancy and key distribution.

3. **Reduce tombstone pressure**
   - Frequent erase + tombstones increases probe lengths and divergence.
   - Run cleanup when tombstoneCounter crosses a threshold, or redesign deletion strategy for your workload.

### `hashinator/hashinator.h`: inlined warp ops (`warpInsert`, `warpInsert_V`, `warpFind`)
Key marked divergence:

- `for (i = 0; i < (1<<sizePower); i += WARPSIZE)` with early break when `warpDone`.
- `while (mask && !warpDone)` with winner lane.
- `if (old == EMPTYBUCKET) ... else if (old == key)`
- `if (threadOverflow > currentMaxBucketOverflow) atomicExch(...)`

#### Classification (warp ops)
- **Mostly inevitable**: these are the fundamental control-flow decisions for open addressing under concurrency.
- **Benign parts**: “winner lane” gating is intentional and reduces atomics.

#### Practical optimization levers
- **Reduce probe length** (most important):
  - Grow earlier (lower load factor), and/or improve hash distribution for structured keys (block indices).
- **Avoid mixed operations** in the same kernel (if possible):
  - E.g., separate “mostly inserts” from “mostly finds” phases to avoid interleaving writes/reads that worsen contention.
- **Contain overflow bounds**:
  - If `_mapInfo->currentMaxBucketOverflow` is large, `warpFind` becomes expensive and divergence increases; keep overflow from growing by resizing earlier.

### `hashinator/hashinator.h`: `clean_tombstones()` overflown-element extraction lambda
Marked around `hashinator.h:1285+`:

- **What it is**: per-bucket lambda `isOverflown(element)` with branches:
  - if `TOMBSTONE` → reset to `EMPTYBUCKET`
  - if `EMPTYBUCKET` → ignore
  - else check whether element is “overflown” vs its optimal hash position
- **Classification**: **inevitable**
  - Tombstones/empties are data-dependent by nature.
  - This is a cleanup path; divergence is usually less important than memory traffic.
- **Optimization levers**:
  - Reduce tombstone creation rate / frequency of erases.
  - Trigger cleanup before tombstones become widespread (so the cleanup kernel touches less “interesting” data).

### `hashinator/hashinator.h`: iterator-style `device_find()` and `insert_element()`
Marked around `hashinator.h:1585+` and `1713+`:

- **What it is**: classic per-thread linear probing with early exits (`continue/return`) and CAS loop for inserts.
- **Classification**: **inevitable** and often **worse than warp-synchronous paths**
  - Thread-level early returns naturally diverge across a warp.
  - CAS loop (`while (i < buckets.size())`) is highly data dependent; some threads succeed early, others spin.
- **Optimization levers**:
  - Prefer warp-synchronous `warpFind`/`warpInsert*` in hot kernels (your code already does this in many places).
  - Avoid calling iterator-style methods from wide GPU kernels unless the expected probe depth is tiny.

---

## Vlasiator GPU: acceleration + sorting + dt + translation

### `vlasovsolver/gpu_acc_map.cpp`

#### `swapBlockIndices(..., dimension)` `switch(dimension)`
Marked at `gpu_acc_map.cpp:51`:

- **Classification**: usually **warp-uniform / benign**
  - `dimension` is a kernel argument, constant across the grid/warp. All lanes take the same `case`.
- **Optimization**: none needed for divergence; keep for readability.

#### `reorder_blocks_by_dimension_kernel`: thread-count guard and column loops
Marked at `gpu_acc_map.cpp:87,98`:

- `if (nThreads != VECL)` is a configuration check.
  - **Classification**: **benign** (should be uniform; ideally always false).
- The loop `for (b = 0; b < columnLength; b++)` is marked.
  - **Important distinction**: loop bounds are **warp-uniform** within a block (one column per block), so this is not “branch divergence” in the classic sense. Cost is proportional to column length, but all lanes iterate equally.

#### Serial kernels: `count_columns_kernel`, `offsets_into_columns_kernel`
Marked at `gpu_acc_map.cpp:156+` and `189+`:

- These are launched with `__launch_bounds__(1,4)` and gated by `(blocki==0 && ti==0)`.
- **Classification**: **not performance relevant** (single-thread/serial by design).
- **Optimization**: if they show up in end-to-end time, the issue is kernel launch overhead or host/device synchronization, not divergence.

#### `evaluate_column_extents_kernel`: shared flag initialization and per-blockK processing
Marked at `gpu_acc_map.cpp:239+` and `418+`:

- Pattern: `for (tti += warpSize) { if (index < MAX_BLOCKS_PER_DIM) { ... } }`
  - **Classification**: **benign** (tail divergence only).
- Later: per-`blockK` conditional actions:
  - `if (isTargetBlock[blockK]!=0) { ... }`
  - `if (isTargetBlock[blockK]!=0 && isSourceBlock[blockK]==0) { device_push_back(...) }`
  - `if (isTargetBlock[blockK]==0 && isSourceBlock[blockK]!=0) { set_element(...) }`
  - **Classification**: **inevitable / data-dependent** (geometry/sparsity).
  - **High-payoff lever** is not removing divergence but avoiding device-side dynamic growth failures:
    - ensure `list_with_replace_new` capacity is sufficient;
    - if out-of-capacity happens, it’s both a divergence site and a correctness/robustness concern.

#### `acceleration_kernel`: the meaningful hot divergence candidates
Marked at `gpu_acc_map.cpp:533,575`:

There are two distinct patterns:

1) Outer `for (k = 0; k < WID * nblocks; ++k)` mark
   - **Classification**: generally **warp-uniform** (all lanes share `nblocks` for a column).

2) Inner `for (gk = minGk; gk <= maxGk; gk++)` mark
   - **This can be a real performance issue if `minGk/maxGk` differ per lane.**
   - Your comment says “Now all w_tids in the warp should have the same gk loop extents”, but they depend on:
     - per-lane `i_indices/j_indices` and sign-based endpoint selection (`intersection_di/dj`),
     - potentially producing different `lagrangian_gk_l/r` per lane.
   - **Classification**:
     - If bounds are truly warp-uniform: divergence mark is **noise**.
     - If bounds vary: **optimizable** (this is one of the few high-payoff divergence fixes).
   - **Optimization pattern (if bounds vary)**:
     - Compute warp-wide bounds:
       - `minGk_warp = warp_min(minGk)`
       - `maxGk_warp = warp_max(maxGk)`
     - Loop `gk` over `[minGk_warp, maxGk_warp]` and predicate each lane’s contribution with `if (gk between lane-min/max)`.
     - This trades predicates for eliminating variable trip counts (often a net win).

Also note the store guard:

- `if (isfinite(tval) && (tval>0) && (tblockLID != invalidLID)) { add }`
  - **Classification**: **inevitable** (physics + sparsity).
  - **Micro-optimization**: order checks cheap→expensive to avoid unnecessary hash lookups:
    - compute `tval`; if `tval <= 0` skip without `getLocalID`.

### `vlasovsolver/gpu_acc_sort_blocks.cpp`

#### `order_GIDs_kernel`: `switch(dimension)` and `if (index < nBlocks)`
Marked at `gpu_acc_sort_blocks.cpp:197+`:

- `switch(dimension)` is typically **warp-uniform** (kernel-constant).
- `if (index < nBlocks)` is a tail guard (last warp).
- **Classification**: mostly **benign**.

#### `construct_columns_kernel`: single-block sanity checks + leader-thread logic
Marked at `gpu_acc_sort_blocks.cpp:269+` and many places inside:

- `if (gpuBlocks != 1) { printf; return; }`
  - **Classification**: **benign** (configuration check).
- Many `if (ti==0)` leader-lane branches:
  - **Classification**: **intentional / benign**.
- “One-column columnset fast path”:
  - `if ((blocksID_mapped_sorted[i+blocks_in_columnset-1] ...)) { skip } else { scan by warp ballot }`
  - **Classification**: **warp-uniform branch** (depends on shared `i` and columnset structure), so not typical warp divergence.
  - **Real cost** is the ballot-based scanning and synchronization, not divergence per se.

**Optimization levers**:
- Reduce device-side `SplitVector::device_push_back` frequency by precomputing counts and reserving.
- Consider building columns on host (if small) or using a two-pass GPU algorithm (count → prefix sum → fill) to avoid dynamic push_back in the main kernel.

### `vlasovsolver/gpu_dt.cpp`: `reduce_v_dt_kernel`
Marked at `gpu_dt.cpp:82`:

- Loop: `for (blockIndex = ti/2; blockIndex < thisVmeshSize; blockIndex += blockSize/2)` with `if (blockIndex < thisVmeshSize)`.
- **Classification**: **benign** (tail divergence only).
- **Dominant costs** are likely:
  - `getGlobalID` + `getBlockInfo` (memory + arithmetic),
  - and reduction synchronization.
- **Optimization levers**:
  - Use warp-level reductions (as suggested in the code comment) to reduce sync overhead.

### `vlasovsolver/gpu_trans_map_amr.cpp`: translation and sparsity

#### `check_skip_remapping(values, vectorindex)`
Marked at `gpu_trans_map_amr.cpp:41`:

- Early exit if any stencil entry is non-zero.
- **Classification**: **beneficial divergence** (sparsity exploitation).
- **Optimization lever (optional)**:
  - Make decision more warp-friendly by voting:
    - each lane checks a subset of stencil entries and uses warp “any” to decide.
  - Only useful if the function itself becomes a hotspot.

#### `translation_kernel`: `switch(dimension)` for `vz_index`
Marked at `gpu_trans_map_amr.cpp:110`:

- **Classification**: **warp-uniform / benign** (dimension is kernel argument).

#### Reading blocks along pencils: `blockLID == invalidLocalID()`
Marked at `gpu_trans_map_amr.cpp:165,172,190,198`:

- **Classification**: **inevitable / data-dependent** (missing blocks due to sparsity / propagation).
- It is also often **beneficial**: avoiding loads and reconstructing zeros.
- **Optimization levers**:
  - If missing blocks are common, consider compacting the list of non-empty cells per pencil to reduce work.

#### Resetting target blocks: `if (pencilRatios[celli] != 0)`
Marked at `gpu_trans_map_amr.cpp:212+`:

- **Classification**: **inevitable** (AMR/sysboundary driven).
- **Optimization**:
  - Precompute and compact “target cell indices” per pencil so reset is over a dense list rather than scanning `sumOfLengths`.

#### Determining blockIndicesD: `if (dimension==0) ... else if ...`
Marked at `gpu_trans_map_amr.cpp:234`:

- **Classification**: **warp-uniform / benign**.

#### Propagation: per-pencil and per-cell conditions
Marked at `gpu_trans_map_amr.cpp:255,258,266,267`:

- `if (pencilBlocksCount[...] == 0) continue;`:
  - **Classification**: data-dependent but usually **warp-uniform** across the block for that pencil (all lanes see same count).
- The store guards:
  - `if (areaRatio && block_data) { ... }`
  - `if (areaRatio_p1 && block_data_p1) { ... }`
  - `if (areaRatio_m1 && block_data_m1) { ... }`
  - **Classification**: **inevitable** (boundary/AMR and missing blocks).

---

## CPU/SIMD reconstruction code (also compiled for device in some paths)

These “warp divergence” marks often represent **SIMD lane divergence** or branchiness in device-compiled scalar paths.

### `vlasovsolver/cpu_1d_ppm_nonuniform.hpp`: PPM monotonicity checks (scalar Realf)
Marked at `cpu_1d_ppm_nonuniform.hpp:82`:

- Ternary/conditional monotonicity corrections for `m_face` and `p_face`.
- **Classification**: **algorithmic/inevitable**
  - This is the limiter/monotonicity logic; removing it changes numerical properties.
- **Optimization**:
  - For vectorized `Vec` paths you already use `select()` which is more SIMD-friendly.
  - For scalar GPU paths, the existing SPF literal fixes (`0.5f`, `1.f/6.f`) help codegen but don’t remove the inherent conditional nature.

### `vlasovsolver/cpu_1d_pqm.hpp`: PQM monotonicity / collapse logic
Marked at `cpu_1d_pqm.hpp:206+` etc. (examples include the `sqrt_val` selection and collapse branches):

- `sqrt_val = (val_to_sqrt < 0) ? (b1 + 200*b2) : sqrt(val_to_sqrt)`
  - **Classification**: **intentional/inevitable** safeguard to avoid invalid sqrt and keep roots outside \[0,1].
- Collapse-to-left vs collapse-to-right and consistency checks:
  - **Classification**: **inevitable** (numerical stability / monotonicity).

### `vlasovsolver/cpu_face_estimates.hpp`: threshold branch in limiter setup
Marked at `cpu_face_estimates.hpp:838`:

- `if (threshold > 0) { scale = 1/threshold; limiter(values*scale); } else { limiter(values); }`
- **Classification**: **warp-uniform / benign** in practice
  - `threshold` is usually a kernel argument / constant; all lanes take same branch.
- **Optimization**:
  - If you want to remove branch entirely: precompute `scale` and always call the same path, but the payoff is likely negligible.

---

## Extra hits outside the original list (present in the repo marks)

The following files also contain `[Warp Divergence]` markers found via search. They are included for completeness.

### `spatial_cell_gpu.cpp`

#### `update_velocity_block_content_lists_kernel`: reduction `if (b_tid < s)`
Marked at `spatial_cell_gpu.cpp:92`:

- Standard reduction pattern; all lanes participate with decreasing active set.
- **Classification**: **benign structural**.
- **Optimization**:
  - Replace with warp-vote reduction (`__ballot_sync` / `__any_sync`) if this kernel is hot and `WID3` fits warp patterns, reducing sync overhead.

#### Map insertion by warp subset
Marked at `spatial_cell_gpu.cpp:99`:

- `if (b_tid < GPUTHREADS) { if (has_content[0]) mapA->warpInsert else mapB->warpInsert }`
- **Classification**: mostly **warp-uniform** (same `has_content[0]` for all lanes); minor divergence from the `b_tid < GPUTHREADS` gating is intentional.

#### `update_velocity_halo_kernel`: `if (newlyadded) { ... }`
Marked at `spatial_cell_gpu.cpp:161`:

- **Classification**: **data-dependent / inevitable** (whether a key was already present).
- **Optimization**:
  - Reduce frequency of redundant inserts by pre-filtering neighbor GIDs (if feasible), but often memory/compute trade-off is unfavorable.

#### `update_neighbour_halo_kernel`: neighbor selection loop
Marked at `spatial_cell_gpu.cpp:199`:

- Loop walks neighbor buffers until locating the right subrange.
- **Classification**: can be **real divergence** (different threads may reach different `neigh_i`).
- **Optimization levers**:
  - Precompute a prefix-sum of neighbor block counts and use binary search (still divergent but fewer steps), or
  - Launch one block per neighbor and avoid mixed neighbor ranges in the same warp.

#### Block adjustment extraction rule lambda
Marked at `spatial_cell_gpu.cpp:1051`:

- `rule_to_replace` compares `kval.second < nBlocksAfterAdjust` etc.
- **Classification**: data-dependent, but likely not the primary cost compared to hashmap extraction itself.

### `velocity_block_container.h`: device push_back capacity check
Marked at `velocity_block_container.h:428`:

- `if (newIndex >= currentCapacityD) assert(0);`
- **Classification**: **benign** if it never triggers; otherwise it indicates a real correctness/capacity bug.
- **Optimization**:
  - Ensure VBC capacity is grown on host before kernels that may add blocks.
  - Avoid device-side resize paths in hot kernels.

### `spatial_cell_gpu.hpp`: `population_increment_kernel` block creation path
Marked at `spatial_cell_gpu.hpp:193+` and inside:

- The code uses warp accessors to find/insert blocks; leader lane (`ti==0`) creates VBC entries.
- The marked `if (!created)` is an error path.
- **Classification**: mostly **benign** (uniform control) with **critical correctness** implications if capacity is insufficient.
- **Optimization**:
  - Pre-reserve vmesh/VBC sizes before increment operations; avoid dynamic creation in the inner loop if possible.

---

## Summary: where divergence is worth optimizing (ranked)

1. **Hashinator probe loops + atomic outcomes**  
   - Don’t try to “remove divergence”; reduce **collisions** and probe depth (resize earlier, better hashing for structured keys).

2. **Any hot loop where per-lane loop bounds differ**  
   - The best candidate in your marks is `gpu_acc_map.cpp` `for(gk=minGk..maxGk)` if `minGk/maxGk` are not truly warp-uniform.

3. **Device-side dynamic push_back/resize under pressure**  
   - Not only divergent, but also a scalability bottleneck and a source of assertion failures. Prefer count→reserve→fill.

4. **Everything else (leader-lane, scans, tails)**  
   - Usually benign; optimize only if they show up at the top of Nsight Compute kernel time or branch-inefficiency.

