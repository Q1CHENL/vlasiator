# Warp divergence markers: `acceleration_kernel` (`vlasovsolver/gpu_acc_map.cpp`)

This file lists all `// [Warp Divergence]` markers:

- Inside `acceleration_kernel` in `vlasovsolver/gpu_acc_map.cpp`
- In device/helper code that `acceleration_kernel` calls or indirectly calls when `ACC_SEMILAG_PQM` is enabled (`vlasovsolver/cpu_1d_pqm.hpp`)

## Direct markers inside `acceleration_kernel`

`acceleration_kernel` begins at line **477** in `vlasovsolver/gpu_acc_map.cpp`.

| Line | Marker location (local context) | Divergence source (as implied by code/comments) |
|---:|---|---|
| 546 | Before the `k` loop over perpendicular slices: `for (uint k=0; k < WID * nblocks; ++k)` | Loop trip count depends on `nblocks` per column; within a thread block it is uniform, but it is annotated as warp divergence. |
| 588 | Before the `gk` loop: `for(int gk = minGk; gk <= maxGk; gk++)` | Loop bounds (`minGk`, `maxGk`) depend on geometry/column extent; comments note this should be uniform within a warp, but bounds can still vary with per-thread geometry. |

## Indirect markers via `compute_pqm_coeff(...)` → `filter_pqm_monotonicity(...)`

When `ACC_SEMILAG_PQM` is defined, `acceleration_kernel` calls:

- `compute_pqm_coeff(...)` (in `vlasovsolver/cpu_1d_pqm.hpp`)
  - which calls `filter_pqm_monotonicity(...)` (same file)

### `filter_pqm_monotonicity(...)` markers (device/Realf + `index` overload)

| Line | Marker location (local context) | What diverges |
|---:|---|---|
| 210 | Conditional sqrt handling: `sqrt_val = (val_to_sqrt < 0) ? ... : sqrt(val_to_sqrt);` | Data-dependent branch on discriminant sign. |
| 256 | Collapse decision: `if (fabs(plm_slope_l) <= fabs(plm_slope_r))` | Data-dependent selection of “collapse to left edge” vs “collapse to right edge”. |
| 269 | Consistency check (left-collapse path): `if (slope_signa * fda_l < 0)` | Data-dependent fix-up branch. |
| 282 | Consistency check (left-collapse path): `else if (slope_signa * fda_r < 0)` | Data-dependent fix-up branch. |
| 308 | Consistency check (right-collapse path): `if (slope_signa * fda_l < 0)` | Data-dependent fix-up branch. |
| 321 | Consistency check (right-collapse path): `else if (slope_signa * fda_r < 0)` | Data-dependent fix-up branch. |

### `compute_pqm_coeff(...)` marker (device/Realf + `index` overload)

| Line | Marker location (local context) | What diverges |
|---:|---|---|
| 360 | Marker appears in the `#ifdef SPF` coefficient assignment block | This is compile-time specialization, but it is annotated as `[Warp Divergence]` in the file. |

