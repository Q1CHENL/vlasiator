# `loop_compact` Warp Divergence Marker Report

This report analyzes every `[Warp Divergence]` marker inside `split::tools::loop_compact` and explains whether it is **true warp divergence** (lane-dependent control flow *within a warp*), whether it is actually **warp-uniform** (all lanes in a warp take the same path), and whether there is a realistic opportunity to reduce it.

## Scope and code under analysis

Kernel analyzed:

```763:848:vlasiator/submodules/hashinator/include/splitvector/split_tools.h
template <typename T, typename Rule, size_t BLOCKSIZE = 1024>
__global__ void loop_compact(split::SplitVector<T, split::split_unified_allocator<T>>& inputVec,
                             split::SplitVector<T, split::split_unified_allocator<T>>& outputVec, Rule rule) {
  // ... kernel body ...
  while (remaining > 0) {
    int current = remaining > blockDim.x ? blockDim.x : remaining;
    __syncthreads();
    // [Warp Divergence]
    const int active = (tid < current) ? rule(input[tid]) : false;
    // ...
    // [Warp Divergence]
    if (wid == 0) {
      // ...
      // [Warp Divergence]
      if (w_tid == 0) {
        // ...
      }
    }
    // [Warp Divergence]
    if (wid == 0) {
      // ...
    }
    // ...
    if (active) { output[warpTidWriteIndex] = input[tid]; }
    // ...
  }
}
```

Notes:
- `WARPLENGTH` is 32 on CUDA and 64 on HIP in this project (`split_tools.h`).
- `wid = tid / WARPLENGTH` is the warp index within the block; `w_tid = tid % WARPLENGTH` is the lane id.

## Marker-by-marker analysis

### Marker 1: `active = (tid < current) ? rule(input[tid]) : false;`

Location:

```787:792:vlasiator/submodules/hashinator/include/splitvector/split_tools.h
while (remaining > 0) {
   int current = remaining > blockDim.x ? blockDim.x : remaining;
   __syncthreads();
   // [Warp Divergence]
   const int active = (tid < current) ? rule(input[tid]) : false;
   const auto mask = split::s_warpVote(active == 1, SPLIT_VOTING_MASK);
```

- **Is it true warp divergence?**: **Yes, potentially**.
  - On the final loop iteration when `current < blockDim.x`, some threads have `tid < current` and others do not. That creates lane-dependent control flow and/or predication.
  - Additionally, even when `tid < current` is uniform, the call to `rule(input[tid])` can itself diverge if `rule` uses data-dependent `if`/early returns.

- **Is there a chance to mitigate?**: **Yes**.
  - **Tail hoisting**: restructure the loop to process all full tiles (`current == blockDim.x`) with no `(tid < current)` check, and handle the final partial tile once. This reduces tail-related divergence to at most one iteration.
  - **Branchless predicates**: rewrite common `rule` implementations to reduce data-dependent branching (convert early-returns into boolean masks when it does not force expensive work for inactive lanes).

### Marker 2: `if (wid == 0) { ... }` (block-wide totalCount reduction step)

Location:

```798:818:vlasiator/submodules/hashinator/include/splitvector/split_tools.h
// Figure out the total here because we overwrite shared mem later
// [Warp Divergence]
if (wid == 0) {
   // ...
   auto localCount = warpSums[w_tid];
   int totalCount = reduceCounts(localCount);
   // [Warp Divergence]
   if (w_tid == 0) {
      outputCount = totalCount;
      outputSize += totalCount;
      assert((outputSize <= capacity) && "loop_compact ran out of capacity!");
      outputVec.device_resize(outputSize);
   }
}
```

- **Is it true warp divergence?**: **No (within a warp)**.
  - `wid` is constant for every lane in a warp, so either **all lanes of warp 0** execute this `if` or **no lanes** (for warps 1..N). That is warp-uniform control flow.

- **Why it can still be slow** (even if not divergent):
  - Only warp 0 does the work; other warps wait at `__syncthreads()`. This is a **warp-0 bottleneck / underutilization** issue, not warp divergence.

- **Is there a chance to mitigate?**: **Yes**, but it’s mainly about reducing warp-0 bottlenecks, not “divergence” per se.
  - Replace the warp-0-centric reduction/scan with a block-wide primitive (so more threads participate) or use a warp-aggregated offset scheme (e.g., per-warp base offsets) to avoid scanning `warpSums` in warp 0.

### Marker 3: `if (w_tid == 0) { ... }` (lane-0 scalar update)

Location:

```810:817:vlasiator/submodules/hashinator/include/splitvector/split_tools.h
int totalCount = reduceCounts(localCount);
// [Warp Divergence]
if (w_tid == 0) {
   outputCount = totalCount;
   outputSize += totalCount;
   assert((outputSize <= capacity) && "loop_compact ran out of capacity!");
   outputVec.device_resize(outputSize);
}
```

- **Is it true warp divergence?**: **Yes (technically)**.
  - `w_tid` differs per lane; only lane 0 executes the body.

- **Is there a chance to mitigate?**: **Usually no meaningful win**.
  - This is a standard pattern to have one lane publish a scalar to shared/global state. The divergent portion is small and often compiled as predication anyway.
  - Any alternative (more lanes doing redundant work, extra synchronization, atomics) commonly costs more than it saves unless the body is heavy.

### Marker 4: `if (wid == 0) { ... }` (prefix scan on `warpSums`)

Location:

```819:830:vlasiator/submodules/hashinator/include/splitvector/split_tools.h
// Prefix scan WarpSums on the first warp
// [Warp Divergence]
if (wid == 0) {
   auto value = warpSums[w_tid];
   for (int d = 1; d < warpsPerBlock; d = 2 * d) {
      int res = split::s_shuffle_up(value, d, SPLIT_VOTING_MASK);
      if (tid % warpsPerBlock >= d) {
         value += res;
      }
   }
   warpSums[w_tid] = value;
}
```

- **Is it true warp divergence?**: **No (for the `wid == 0` itself)**, for the same reason as Marker 2.
  - Again, the “divergence” is across warps (warp 0 active, other warps idle), not within a warp.

- **But there is real intra-warp divergence inside the scan**:
  - The inner `if (tid % warpsPerBlock >= d)` is lane-dependent (classic scan implementation). That is true intra-warp divergence/predication *inside warp 0*.

- **Is there a chance to mitigate?**: **Some**, but typically this is not the highest-impact divergence source compared to Marker 1 and the compaction write.
  - You can replace the scan approach (block-wide scan primitive, different offset scheme), but that changes the implementation structure.

## Important divergence that is unmarked (but real)

This is the core compaction divergence and is inherent to stream compaction:

```832:837:vlasiator/submodules/hashinator/include/splitvector/split_tools.h
const auto warpTidWriteIndex = offset + pp;
if (active) {
   output[warpTidWriteIndex] = input[tid];
}
```

- **Is it true warp divergence?**: **Yes** — different lanes have different `active` values.
- **Is there a chance to mitigate?**: **Not in the sense of eliminating it**.
  - Stream compaction necessarily performs predicated writes.
  - What *can* be mitigated is the divergence that leads to `active` (Marker 1 tail divergence and divergence inside `rule`).

## Practical summary (divergence-focused)

- **Highest-value divergence to reduce**: Marker 1 (tail-related `tid < current` branch) via **tail hoisting**; and divergence inside `rule(...)` via more branchless predicates where applicable.
- **Markers 2 and 4 are misclassified as warp divergence**: they are mostly **warp-uniform control flow** but can still be performance bottlenecks due to **warp-0-only work + synchronization**.
- **Marker 3 is real but small**: single-lane scalar updates rarely dominate.

