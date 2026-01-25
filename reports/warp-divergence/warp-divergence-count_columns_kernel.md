# Warp divergence markers: `count_columns_kernel` (`vlasovsolver/gpu_acc_map.cpp`)

This file lists all `// [Warp Divergence]` markers inside `count_columns_kernel` in `vlasovsolver/gpu_acc_map.cpp`.

`count_columns_kernel` begins at line **153** in `vlasovsolver/gpu_acc_map.cpp`.

| Line | Marker location (local context) | Divergence source (as implied by code/comments) |
|---:|---|---|
| 168 | Guarding the serial work: `if ((blocki==0)&&(ti==0)) { ... }` | A single lane (thread 0 of block 0) performs the counting work while other threads do nothing. |
| 170 | Immediately before the outer loop: `for(uint setIndex=0; setIndex< gpu_columnData->setColumnOffsets.size(); ++setIndex)` | Loop is executed only under the `(blocki==0)&&(ti==0)` guard; no other `[Warp Divergence]` markers are present inside this kernel. |

