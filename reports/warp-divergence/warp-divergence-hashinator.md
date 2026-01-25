# Warp divergence markers: `Hashinator` (`submodules/hashinator/include/hashinator/hashinator.h`)

This file lists all `// [Warp Divergence]` markers in `submodules/hashinator/include/hashinator/hashinator.h`.

Note: this is **not a single kernel**; it includes device-side warp-cooperative routines and other GPU-facing code paths.

## Marker inventory (grouped by function/region)

### `warpInsert(...)`

| Line | Context |
|---:|---|
| 882 | Winner-lane only updates on duplicate detection (`if (w_tid == winner)`) |
| 897 | Winner-lane attempts insertion into an empty bucket region (`if (w_tid == winner)`) |
| 900 | Conditional based on CAS result (`if (old == EMPTYBUCKET)`) |
| 906 | Conditional update of `currentMaxBucketOverflow` (`if (threadOverflow > _mapInfo->currentMaxBucketOverflow)`) |
| 911 | CAS edge-case branch (`else if (old == candidateKey)`) |

### `warpInsert_V(...)`

| Line | Context |
|---:|---|
| 953 | Outer probing loop over table size (`for (size_t i = 0; i < (1 << sizePower); i += defaults::WARPSIZE)`) |
| 984 | Inner loop over empties mask (`while (mask && !warpDone)`) |
| 987 | Winner-lane CAS (`if (w_tid == winner)`) |
| 995 | Local count / fill update executed only on successful insert path |
| 997 | Conditional update of `currentMaxBucketOverflow` (`if (threadOverflow > _mapInfo->currentMaxBucketOverflow)`) |
| 1002 | CAS edge-case branch (`else if (old == candidateKey)`) |

### `warpFind(...)`

| Line | Context |
|---:|---|
| 1043 | Probing loop bounded by `currentMaxBucketOverflow` |
| 1060 | Branch when `maskExists` is true |
| 1063 | Winner-lane reads candidate value (`if (w_tid == winner)`) |

### `warpErase(...)`

| Line | Context |
|---:|---|
| 1095 | Probing loop bounded by `currentMaxBucketOverflow` |
| 1112 | Branch when `maskExists` is true |
| 1115 | Winner-lane writes tombstone (`if (w_tid == winner)`) |
| 1118 | Winner-lane atomic updates of tombstone/fill counters |

### Cleanup lambda `isOverflown` (overflow handling)

| Line | Context |
|---:|---|
| 1291 | Branch removing tombstones (`if (element.first == TOMBSTONE)`) |
| 1296 | Branch skipping empty buckets (`if (element.first == EMPTYBUCKET)`) |
| 1304 | Data-dependent overflow classification (`bool isOverflown = (...)`) |

### `device_find(...)`

| Line | Context |
|---:|---|
| 1585 | Linear probe loop (`for (size_t i = 0; i < _mapInfo->currentMaxBucketOverflow; i++)`) |
| 1588 | Skip tombstones (`if (candidate.first == TOMBSTONE)`) |
| 1592 | Found-key early return (`if (candidate.first == key)`) |
| 1597 | Empty-bucket early return (`if (candidate.first == EMPTYBUCKET)`) |
| 1600 | Return `device_end()` on empty bucket |

### `insert_element(...)`

| Line | Context |
|---:|---|
| 1714 | Probe loop (`while (i < buckets.size())`) |
| 1719 | Empty-bucket insert case (`if (old == EMPTYBUCKET)`) |
| 1729 | Existing-key overwrite case (`if (old == key)`) |
| 1732 | Updates `thread_overflowLookup` in overwrite path |
| 1739 | Hard failure when table overflows (`assert(false && "Hashmap completely overflown")`) |

