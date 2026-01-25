# Warp divergence report: `loop_compact_keys` (`submodules/hashinator/include/splitvector/split_tools.h`)

This report covers:

- The CUDA kernel `split::tools::loop_compact_keys(...)` in `submodules/hashinator/include/splitvector/split_tools.h`
- The kernel launch site (`copy_if_keys_loop(...)`)
- Any `[Warp Divergence]` markers in code directly invoked by this kernel (notably `SplitVector::device_resize(...)` in `splitvec.h`)

## 1) Entry points and call chain

### Host launch site

`copy_if_keys_loop(...)` launches the kernel:

- File: `submodules/hashinator/include/splitvector/split_tools.h`
- Launch: `split::tools::loop_compact_keys<<<1, BLOCKSIZE, 0, s>>>(input, output, rule);` (around line **1072**)

### Kernel

- Kernel: `__global__ void loop_compact_keys(...)` (begins around line **850** in `split_tools.h`)
- Calls (relevant to divergence markers):
  - `rule(input[tid])` (user-provided predicate; data-dependent)
  - `outputVec.device_resize(outputSize)` (device method; has `[Warp Divergence]` markers)
  - warp primitives (e.g., `split::s_warpVote`, shuffles); no `[Warp Divergence]` markers were traced in their definitions as part of this request

## 2) `[Warp Divergence]` markers inside `loop_compact_keys`

File: `submodules/hashinator/include/splitvector/split_tools.h`

| Line | Marker location (local context) | Why it diverges |
|---:|---|---|
| 876 | `const int active = (tid < current) ? rule(input[tid]) : false;` | The final iteration of the `while (remaining > 0)` loop can have `current < blockDim.x`, so some threads are inactive (`tid >= current`). Also `rule(...)` is data-dependent. |
| 885 | `if (wid == 0) { ... }` (reduce `warpSums` to `totalCount`) | Only the first warp (where `wid==0`) performs the reduction; other warps do not execute this block. |
| 897 | `if (w_tid == 0) { outputCount = totalCount; ... outputVec.device_resize(outputSize); }` | Single-lane control path (lane 0 of warp 0) updates shared counters and resizes output. |
| 906 | `if (wid == 0) { ... }` (prefix-scan `warpSums`) | Again, only warp 0 performs the prefix-scan; other warps do not. |

### Note: unmarked but real divergence in this kernel

Even where not annotated, these branches can still diverge:

- `if (active) { output[warpTidWriteIndex] = input[tid].first; }` (data-dependent on `rule` and on the partial-tile condition `tid < current`)
- `while (remaining > 0)` iterates until exhaustion; all threads follow the same loop condition, but useful work differs on the last iteration.

## 3) Traced markers in code called by `loop_compact_keys`

### `SplitVector::device_resize(...)`

File: `submodules/hashinator/include/splitvector/splitvec.h`

`loop_compact_keys` calls `outputVec.device_resize(outputSize)` (from lane 0 of warp 0). The device resize implementation contains these markers:

| Line | Marker location (local context) | Why it diverges |
|---:|---|---|
| 913 | `if (newSize > capacity()) { assert(0 && "..."); }` | Conditional failure path; typically uniform but still a branch (and a catastrophic error if triggered). |
| 918 | `if (construct) { for (size_t i = size(); i < newSize; ++i) { ... } }` | Branch and loop depend on `construct` and on `newSize - size()`. In `loop_compact_keys` this is called only by one thread, so “warp divergence” here mostly reflects single-thread control flow rather than lane-level divergence. |

## 4) Assessment (what matters)

- **Main divergence driver**: `rule(input[tid])` and the resulting `active` mask. This is fundamental to compaction: if many lanes fail the predicate, the kernel naturally becomes control/compute imbalanced.
- **Structural divergence**: the `wid == 0` and `w_tid == 0` paths intentionally concentrate bookkeeping work in warp 0 / lane 0. This is usually a good tradeoff (less synchronization, fewer atomics), but it does mean most lanes idle during reduction/resize steps.
- **Last-iteration effect**: the `(tid < current)` condition is only “bad” on the final chunk when `remaining < blockDim.x`. If inputs are typically large, this divergence is amortized; if inputs are frequently small or near `BLOCKSIZE`, it becomes more significant.
- **Correctness risk**: `device_resize` asserts if `outputSize > capacity`. That’s a hard failure; if you see this in practice, the important action is ensuring `outputVec` is pre-reserved to a safe upper bound before calling `copy_if_keys_loop`.

