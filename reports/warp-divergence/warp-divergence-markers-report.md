# Warp divergence markers report

This report inventories all `// [Warp Divergence]` markers requested in:

- `vlasovsolver/gpu_acc_map.cpp`: **only** those inside `acceleration_kernel`
- `vlasovsolver/cpu_1d_pqm.hpp`: markers in the PQM path that `acceleration_kernel` reaches when `ACC_SEMILAG_PQM` is enabled
- `submodules/hashinator/include/hashinator/hashinator.h`: all markers in that header (included by request; see note below)

Extraction method: literal search for the string `"[Warp Divergence]"` and then reading local code context around each match.

## Call-path context (why these files matter)

- `vlasovsolver/gpu_acc_map.cpp` defines the CUDA kernel `acceleration_kernel(...)`.
- Inside `acceleration_kernel`, when `ACC_SEMILAG_PQM` is defined, it calls:
  - `compute_pqm_coeff(...)` (declared in `vlasovsolver/cpu_1d_pqm.hpp`)
    - which calls `filter_pqm_monotonicity(...)` (also in `cpu_1d_pqm.hpp`)

Note: `acceleration_kernel` itself does **not** call into `Hashinator` APIs. However, `gpu_acc_map.cpp` (outside `acceleration_kernel`) uses `Hashinator::Hashmap`/`hash_pair` types in other kernels and orchestration code, so `hashinator.h` may still be part of the larger GPU acceleration workflow even if it is not in the direct `acceleration_kernel` call chain.

## `vlasovsolver/gpu_acc_map.cpp` — `acceleration_kernel` markers

`acceleration_kernel` begins at line **477** in this file. Only **two** `[Warp Divergence]` markers occur inside the kernel body:

| Line | Marker location (local context) | Divergence source (as implied by code/comments) |
|---:|---|---|
| 546 | Before the `k` loop over perpendicular slices: `for (uint k=0; k < WID * nblocks; ++k)` | Loop trip count depends on `nblocks` per column; this can vary by column, but within a block it is uniform. Marker likely reflects loop structure / earlier GPU-scout annotation. |
| 588 | Before the `gk` loop: `for(int gk = minGk; gk <= maxGk; gk++)` | Loop bounds (`minGk`, `maxGk`) depend on geometry and column extent; comments note intent to make extents uniform within warp, but bounds can still vary across threads depending on how `j_indices`/`intersection_*` affect `lagrangian_gk_*`. |

## `vlasovsolver/cpu_1d_pqm.hpp` — PQM monotonicity / coefficient markers

These markers are reached from `acceleration_kernel` when `ACC_SEMILAG_PQM` is compiled in, because `acceleration_kernel` calls `compute_pqm_coeff(...)` which calls `filter_pqm_monotonicity(...)`.

### Markers in `filter_pqm_monotonicity(...)` (Realf/index version used on device)

| Line | Marker location (local context) | What diverges |
|---:|---|---|
| 210 | Conditional sqrt handling: `sqrt_val = (val_to_sqrt < 0) ? ... : sqrt(val_to_sqrt);` | Data-dependent branch on discriminant sign to avoid `sqrt(negative)` and force roots outside \([0,1]\). |
| 256 | Collapse decision: `if (fabs(plm_slope_l) <= fabs(plm_slope_r))` | Data-dependent selection of “collapse to left edge” vs “collapse to right edge”. |
| 269 | Consistency check (left-collapse path): `if (slope_signa * fda_l < 0)` | Data-dependent fix-up branch for slope consistency. |
| 282 | Consistency check (left-collapse path): `else if (slope_signa * fda_r < 0)` | Data-dependent alternate fix-up branch for slope consistency. |
| 308 | Consistency check (right-collapse path): `if (slope_signa * fda_l < 0)` | Data-dependent fix-up branch for slope consistency. |
| 321 | Consistency check (right-collapse path): `else if (slope_signa * fda_r < 0)` | Data-dependent alternate fix-up branch for slope consistency. |

### Marker in `compute_pqm_coeff(...)` (Realf/index device version)

| Line | Marker location (local context) | What diverges |
|---:|---|---|
| 360 | Marker appears in the `#ifdef SPF` float-precision coefficient assignment block | This is not an obvious control-flow divergence by itself (it is compile-time), but it is annotated as `[Warp Divergence]` in the file. |

## `submodules/hashinator/include/hashinator/hashinator.h` — markers

This header contains **31** `[Warp Divergence]` markers, primarily in warp-cooperative insert/find/erase routines and in device-side linear probing.

### `warpInsert(...)` markers

| Line | Context |
|---:|---|
| 882 | Winner-lane only updates on duplicate detection (`if (w_tid == winner)`) |
| 897 | Winner-lane attempts insertion into an empty bucket region (`if (w_tid == winner)`) |
| 900 | Conditional based on CAS result (`if (old == EMPTYBUCKET)`) |
| 906 | Conditional update of `currentMaxBucketOverflow` (`if (threadOverflow > _mapInfo->currentMaxBucketOverflow)`) |
| 911 | CAS edge-case branch (`else if (old == candidateKey)`) |

### `warpInsert_V(...)` markers

| Line | Context |
|---:|---|
| 953 | Outer probing loop over table size (`for (size_t i = 0; i < (1 << sizePower); i += defaults::WARPSIZE)`) |
| 984 | Inner loop over empties mask (`while (mask && !warpDone)`) |
| 987 | Winner-lane CAS (`if (w_tid == winner)`) |
| 995 | Local count / fill update executed only on successful insert path |
| 997 | Conditional update of `currentMaxBucketOverflow` (`if (threadOverflow > _mapInfo->currentMaxBucketOverflow)`) |
| 1002 | CAS edge-case branch (`else if (old == candidateKey)`) |

### `warpFind(...)` markers

| Line | Context |
|---:|---|
| 1043 | Probing loop bounded by `currentMaxBucketOverflow` |
| 1060 | Branch when `maskExists` is true |
| 1063 | Winner-lane reads candidate value (`if (w_tid == winner)`) |

### `warpErase(...)` markers

| Line | Context |
|---:|---|
| 1095 | Probing loop bounded by `currentMaxBucketOverflow` |
| 1112 | Branch when `maskExists` is true |
| 1115 | Winner-lane writes tombstone (`if (w_tid == winner)`) |
| 1118 | Winner-lane atomic updates of tombstone/fill counters |

### Cleanup lambda `isOverflown` markers (used during overflow handling)

| Line | Context |
|---:|---|
| 1291 | Branch removing tombstones (`if (element.first == TOMBSTONE)`) |
| 1296 | Branch skipping empty buckets (`if (element.first == EMPTYBUCKET)`) |
| 1304 | Data-dependent overflow classification (`bool isOverflown = (...)`) |

### `device_find(...)` markers

| Line | Context |
|---:|---|
| 1585 | Linear probe loop (`for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++)`) |
| 1588 | Skip tombstones (`if (candidate.first == TOMBSTONE)`) |
| 1592 | Found-key early return (`if (candidate.first == key)`) |
| 1597 | Empty-bucket early return (`if (candidate.first == EMPTYBUCKET)`) |
| 1600 | Return `device_end()` on empty bucket |

### `insert_element(...)` markers

| Line | Context |
|---:|---|
| 1714 | Probe loop (`while (i < buckets.size())`) |
| 1719 | Empty-bucket insert case (`if (old == EMPTYBUCKET)`) |
| 1729 | Existing-key overwrite case (`if (old == key)`) |
| 1732 | Updates `thread_overflowLookup` in overwrite path |
| 1739 | Hard failure when table overflows (`assert(false && "Hashmap completely overflown")`) |

