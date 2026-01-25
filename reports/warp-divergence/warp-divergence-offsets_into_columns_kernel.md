# Warp divergence report: `offsets_into_columns_kernel` (`vlasovsolver/gpu_acc_map.cpp`)

This report lists all `// [Warp Divergence]` markers inside `offsets_into_columns_kernel` in `vlasovsolver/gpu_acc_map.cpp`.

`offsets_into_columns_kernel` begins at line **190** in `vlasovsolver/gpu_acc_map.cpp`.

## Marker inventory (inside the kernel)

| Line | Marker location (local context) | Divergence source (as implied by code/comments) |
|---:|---|---|
| 201 | Immediately before outer loop: `for (uint setIndex=0; setIndex < gpu_columnData->setColumnOffsets.size(); ++setIndex)` | Loop executes only under `if ((blocki==0)&&(ti==0))` (serialized single-thread kernel usage). |
| 204 | Immediately before inner loop over columns in the set | Still under the same single-thread guard; marker likely reflects loop structure rather than lane-level divergence. |
| 208 | Before bounds check: `if (valuesColumnOffset >= valuesSizeRequired)` | Data-dependent branch for overflow detection (prints an error). |

## Assessment

- **This kernel is effectively serialized**: it runs under `__launch_bounds__(1,4)` and explicitly gates work to `if ((blocki==0)&&(ti==0))`, so almost all “warp divergence” here is not warp-lane divergence in the usual sense—it's “only one thread does work.”
- **Only data-dependent branch**: the overflow check at line **208**; if it triggers, it indicates a correctness/configuration problem (values array too small) rather than a performance tradeoff.

