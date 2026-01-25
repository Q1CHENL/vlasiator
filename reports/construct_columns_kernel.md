# What `construct_columns_kernel` does (Vlasiator GPU acceleration “columns”)

This note explains the *semantics* of `construct_columns_kernel` in
`vlasovsolver/gpu_acc_sort_blocks.cpp` (kernel around lines ~257–437).

**One-sentence summary:** it scans the **sorted mapped block keys** and builds offset/length metadata that groups blocks into **contiguous “columns”** (runs along the chosen dimension) and into **column sets** (same `column_id`) for downstream acceleration kernels.

It is meant to be read after `reports/vlasiator-in-detail.md`, and it complements (does not replace)
the marker-focused writeup in:

- `reports/warp-divergence/warp-divergence-construct_columns_kernel.md`

---

## Where this kernel sits in the pipeline

The surrounding function `sortBlocklistByDimension(...)` does (on GPU):

1. **Map each active block** (per spatial cell + population) into a 1D sortable key where the chosen dimension
   becomes the “fastest varying” axis (`blocksID_mapped_dim{0,1,2}_kernel`).
2. **Radix sort** the mapped keys, carrying the corresponding block `LID`s (CUB/HIPCUB `SortPairs`).
3. **Reorder `GID`s and count blocks per “column id”** (`order_GIDs_kernel`):
   - `blocksGID[index] = vmesh->getGlobalID(blocksLID[index])`
   - `gpu_columnNBlocks[column_id]++` via `atomicAdd`
4. **Construct column metadata** (`construct_columns_kernel`):
   - emits offsets/lengths describing contiguous runs (“columns”) in the sorted list
   - also groups those columns into “column sets” per `column_id`

This report is about step (4).

---

## Inputs and outputs (what the kernel consumes/produces)

## Kernel signature and parameter-by-parameter meaning (no skips)

The kernel is defined as:

```cpp
__global__ void construct_columns_kernel(
   const vmesh::VelocityMesh* vmesh,
   const uint dimension,
   vmesh::GlobalID *blocksID_mapped_sorted,
   vmesh::LocalID *gpu_columnNBlocks,
   ColumnOffsets* columnData,
   const uint nBlocks
);
```

Below is what **each parameter** means, including what is expected on entry and what is produced/consumed.

### `const vmesh::VelocityMesh* vmesh`

- **What it is**: a device pointer to the per-(spatial cell, population) velocity mesh object that provides mesh geometry
  (notably `gridLength[3]`) and mappings.
- **What this kernel uses it for**: only to compute:
  - `DX = vmesh->getGridLength()[dimension]`
- **Expectations**:
  - `vmesh` must be valid on device for the duration of the kernel.
  - `vmesh->getGridLength()` must return a pointer/array with at least 3 entries.
  - `dimension` must be in-range so the indexing is valid.

### `const uint dimension`

- **What it is**: which velocity axis is treated as the “column direction”.
- **Allowed values**: `0`, `1`, or `2`.
- **How it is used**:
  - selects `DX = gridLength[dimension]`
  - defines how `blocksID_mapped_sorted[i]` is split into:
    - `column_id = mapped / DX`
    - `dimension_id = mapped % DX`
- **Expectation**: must match the mapping used earlier when generating/sorting `blocksID_mapped(_sorted)`.

### `vmesh::GlobalID* blocksID_mapped_sorted`

- **What it is**: a device pointer to an array of length `nBlocks`.
- **What it contains on entry**: the **sorted mapped IDs** (keys) produced by the radix sort stage.
  - These are *not* raw `GID`s. They are “dimension-reordered” IDs designed so adjacency along `dimension` corresponds to
    adjacency in this key space.
- **How it is used**:
  - scanned from `i=0..nBlocks-1` to detect boundaries between:
    - different `column_id`s (new “column set”), and
    - gaps in `dimension_id` (new “column segment” within the same `column_id`)
  - used for the fast-path lookahead at index `i + blocks_in_columnset - 1`.
- **Expectations / hazards**:
  - Must be sorted ascending.
  - Must be grouped by `column_id` so that all elements with the same `column_id` occur in one contiguous region.
  - The kernel can read `blocksID_mapped_sorted[i + blocks_in_columnset - 1]`; therefore `blocks_in_columnset` (derived from
    `gpu_columnNBlocks`) must be correct, otherwise this access can go out-of-bounds.

### `vmesh::LocalID* gpu_columnNBlocks`

- **What it is**: a device pointer to an array indexed by `column_id`.
- **What it contains on entry**: for each `column_id`, the number of blocks in the sorted list belonging to that `column_id`.
  - This is computed earlier (in `order_GIDs_kernel`) by doing `atomicAdd(&gpu_columnNBlocks[column_id], 1)` for each block.
- **How it is used**:
  - at the start of processing a `column_id`, the leader thread loads:
    - `blocks_in_columnset = gpu_columnNBlocks[column_id]`
  - `blocks_in_columnset` then drives:
    - the fast-path “skip whole column set” check
    - the slow-path scan limit
    - the decremented “how many remain in this set” bookkeeping after consuming one segment
- **Expectations**:
  - The array must be **zeroed/reset** before counting begins, otherwise counts accumulate across invocations.
  - Counts must match the grouping in `blocksID_mapped_sorted` (same `dimension`, same mapping scheme).

### `ColumnOffsets* columnData`

- **What it is**: a device pointer to a structure that owns four device-side dynamic arrays (SplitVector-like).
- **What it must be on entry**: logically “empty” (cleared) so that this kernel can append offsets/lengths from scratch.
  - This kernel does not clear; it only does `device_push_back(...)`.
- **What this kernel writes** (appends):
  - `columnData->columnBlockOffsets`: start indices into `blocksID_mapped_sorted` for each column segment
  - `columnData->columnNumBlocks`: lengths (in blocks) for each column segment
  - `columnData->setColumnOffsets`: start indices into `columnBlockOffsets` for each column set (each `column_id` region)
  - `columnData->setNumColumns`: number of column segments within each column set
- **Additional expectation**:
  - `device_push_back` and `.size()` must be safe to call from device code in this context.
  - In this kernel, pushes are done by the leader thread (`ti==0`) to keep ordering deterministic.

### `const uint nBlocks`

- **What it is**: the length of `blocksID_mapped_sorted`, i.e. the number of **active sparse velocity blocks** in the
  current (spatial cell, population) being processed.
- **How it is used**:
  - loop upper bound `while (i < nBlocks)`
  - final-length computation for the last column segment
- **Expectations**:
  - Must match the allocations and valid range of `blocksID_mapped_sorted`.

### Inputs

`construct_columns_kernel(...)` takes:

- `dimension`: which velocity axis is treated as the “column direction” (0/1/2)
- `DX = vmesh->getGridLength()[dimension]`: number of possible block positions along that axis
- `blocksID_mapped_sorted[0..nBlocks-1]`:
  - the mapped IDs after radix sort (ascending)
  - **key property**: for a fixed “column”, mapped IDs increase by +1 as you advance by one block along `dimension`
- `gpu_columnNBlocks[column_id]`:
  - for each `column_id`, how many blocks exist in the sparse list for that column_id
  - this is computed in `order_GIDs_kernel` by `atomicAdd`

### Outputs: `ColumnOffsets* columnData`

The kernel fills four device-side dynamic arrays (a `SplitVector`-like container with `device_push_back`):

- `columnBlockOffsets`: **start index** (into `blocksID_mapped_sorted`) where each *column segment* begins
- `columnNumBlocks`: **length** (#blocks) of each column segment
- `setColumnOffsets`: **start index** (into `columnBlockOffsets`) where each *column set* begins
- `setNumColumns`: **length** (#columns) of each column set

You can treat the outputs as a two-level segmentation:

- **Level 1 (columns):** contiguous runs of blocks that are consecutive along `dimension`
- **Level 2 (column sets):** groups of those columns that share the same `(other two coordinates)` i.e. the same `column_id`

---

## Key concept: “mapped id”, `column_id`, and `dimension_id`

The kernel interprets each sorted mapped id `m = blocksID_mapped_sorted[i]` as:

- `column_id = m / DX`
- `dimension_id = m % DX`

Interpretation:

- `column_id` identifies the “column” in the *plane orthogonal to `dimension`* (i.e. the other two velocity-block indices).
- `dimension_id` is the block index along the chosen `dimension` within that column.

Because the mapped keys were designed so that `dimension_id` is the “fastest varying” coordinate, blocks that are
adjacent along the column direction have adjacent mapped IDs.

---

## What counts as a “column” here?

In this kernel, a **column** is **not** “all blocks with the same `column_id`”.

Instead, a **column (segment)** is a *maximal contiguous run* in the sorted list where:

- `column_id` stays the same, and
- `dimension_id` increases by exactly 1 from one element to the next.

This matters because the sparse block set for one `column_id` may have **gaps** along `dimension`:

- thresholding / sparse activation can remove blocks in the middle
- multiple populations (or other logic) can create “breaks”

Those gaps are treated as **separate columns (segments)**, even though they share the same `column_id`.

The term used in comments is:

- **“column set”** = all blocks belonging to one `column_id`
- **“columns”** = the contiguous segments *within* that column_id, split by gaps

---

## Execution model: single-block, leader-driven scan

The kernel is launched as `<<<1, GPUTHREADS>>>` and explicitly rejects other grid sizes:

- It computes `gpuBlocks = gridDim.x * gridDim.y * gridDim.z`
- If `gpuBlocks != 1`, it prints an error and returns

So the algorithm is essentially:

- one CUDA/HIP thread block
- shared loop index `i` (leader thread updates it)
- all threads cooperate mainly for the ballot-based scan in the “gap” case

This is why many `[]` markers in the code refer to “intentional divergence/serialization”.

---

## Step-by-step algorithm (exactly what happens)

### Initialization (leader thread)

Shared state:

- `__shared__ vmesh::LocalID i;`
- `__shared__ vmesh::LocalID blocks_in_columnset;`

Leader (`ti==0`) initializes:

- `i = 0`
- `blocks_in_columnset = 0`
- `columnBlockOffsets.push_back(0)` (first column starts at 0)
- `setColumnOffsets.push_back(0)` (first column set starts at column index 0)

### Main loop over the sorted list

The kernel runs:

```text
while (i < nBlocks) { ... }
```

At the top of each iteration it computes:

- `column_id = blocksID_mapped_sorted[i] / DX`
- `dimension_id = blocksID_mapped_sorted[i] % DX`

Then:

1) **Load the total size of the current column set (once per set).**

If `blocks_in_columnset == 0`, the leader loads:

- `blocks_in_columnset = gpu_columnNBlocks[column_id]`

This is the number of *remaining* blocks associated with the current `column_id`.

2) **Detect whether a new column segment starts at `i`.**

For `i > 0`, the leader compares the current pair `(column_id, dimension_id)` to the previous iteration’s values
`(prev_column_id, prev_dimension_id)` and starts a new column if either:

- `column_id != prev_column_id` (new column set), or
- `dimension_id != prev_dimension_id + 1` (gap along the dimension)

When a new column is detected, it appends:

- `columnBlockOffsets.push_back(i)` (start of the new column)
- `columnNumBlocks.push_back(length_of_previous_column)`
  - computed as the difference of the last two `columnBlockOffsets`

If `column_id` changed (i.e. a new column set began), it also appends:

- `setColumnOffsets.push_back(index_of_current_column)`
  - implemented as `columnBlockOffsets.size() - 1`
- `setNumColumns.push_back(num_columns_in_previous_set)`
  - difference of last two `setColumnOffsets`

3) **Fast-path: if the entire remaining column set is one contiguous run, skip it.**

The kernel checks the *last block in the current column set* (relative to current `i`):

- look at `blocksID_mapped_sorted[i + blocks_in_columnset - 1]`
- verify:
  - same `column_id`
  - `dimension_id_last == dimension_id + blocks_in_columnset - 1`

If true, then the remaining blocks for this `column_id` form one contiguous sequence starting at the current
`dimension_id`. The leader does:

- `i += blocks_in_columnset`
- `blocks_in_columnset = 0`

and the loop continues (the next iteration will start a new column/set as needed).

4) **Slow-path: gaps exist; find the length of the current contiguous run using warp ballots.**

If the fast-path check fails, the current column set contains one or more gaps, so the kernel must determine
how long the **current** column segment (starting at `i`) is.

It does this in chunks of `warpSize` elements (where `warpSize = blockDim.x*blockDim.y*blockDim.z`).
For each chunk starting at offset `ci` within the remaining column set:

- Each lane `ti` (if `ci + ti < blocks_in_columnset`) checks whether element `i + ci + ti` still belongs to the same
  contiguous run:
  - same `column_id`, and
  - `dimension_id_at_element == dimension_id + (ci + ti)`
- Lanes vote with a ballot on `notInColumn` (1 means “this element breaks the run”).
- `__ffs(ballot_result)` finds the first lane that reported “break”.
  - If no lane reports a break, the chunk is fully in the column segment, and the run can extend by `warpSize`.
  - Otherwise, the run ends inside this chunk.

Once a break is found:

- `this_col_length` is the run length discovered so far
- the leader advances:
  - `i += this_col_length`
  - `blocks_in_columnset -= this_col_length`

Crucially, after consuming one segment, the kernel continues within the same `column_id` (same column set),
but `blocks_in_columnset` now represents “how many blocks remain in this column set after the segment we just consumed”.

### Finalization (leader thread)

After `i` reaches `nBlocks`, the kernel appends the final segment lengths:

- `columnNumBlocks.push_back(nBlocks - last_column_offset)`
- `setNumColumns.push_back(total_columns - last_set_offset)`

This closes the last open column and the last open column set.

---

## Concrete example (small numbers)

This example shows the full chain:

- start from a sparse set of active velocity blocks (3D indices)
- compute a “normal” flattened `GID`
- compute a **mapped** key where the chosen `dimension` is fastest-varying
- **sort** by that mapped key (what CUB/HIPCUB does)
- show how `construct_columns_kernel` interprets the sorted mapped keys into `(column_id, dimension_id)` and thus columns

Assume the velocity-block grid extents are:

- `D0=2` blocks in x, `D1=3` blocks in y, `D2=2` blocks in z

So valid block coordinates are $(x,y,z)$ with:

- $x \in \{0,1\}$
- $y \in \{0,1,2\}$
- $z \in \{0,1\}$

### Step A: pick a sparse set of active blocks

Assume only these 5 blocks exist (in one spatial cell + population):

- A: $(0,1,0)$
- B: $(1,1,0)$
- C: $(0,2,0)$
- D: $(0,0,1)$
- E: $(0,1,1)$

### Step B: compute their “normal” flattened `GID` (one possible linearization)

One common flattening (x fastest-varying) is:

$$
GID = x + D0\cdot(y + D1\cdot z)
$$

Compute:

- A $(0,1,0)$: $GID=0 + 2(1+3\cdot0)=2$
- B $(1,1,0)$: $GID=1 + 2(1)=3$
- C $(0,2,0)$: $GID=0 + 2(2)=4$
- D $(0,0,1)$: $GID=0 + 2(0+3)=6$
- E $(0,1,1)$: $GID=0 + 2(1+3)=8$

(The actual sparse storage order is arbitrary; this is just the “global identity”.)

### Step C: choose `dimension = 1` (y) and compute a “mapped” key (y fastest-varying)

To make **y** fastest-varying, you choose a different linearization such that incrementing y increments the key by 1.
One such mapping is:

$$
mapped = y + D1\cdot(x + D0\cdot z)
$$

Compute mapped keys:

- A $(0,1,0)$: $mapped=1 + 3(0+2\cdot0)=1$
- B $(1,1,0)$: $mapped=1 + 3(1)=4$
- C $(0,2,0)$: $mapped=2 + 3(0)=2$
- D $(0,0,1)$: $mapped=0 + 3(0+2)=6$
- E $(0,1,1)$: $mapped=1 + 3(0+2)=7$

Now form the “sort pairs” input (key, value):

- keys: `blocksID_mapped = [1, 4, 2, 6, 7]`
- values (carried along): `blocksLID_unsorted = [LID(A), LID(B), LID(C), LID(D), LID(E)]`

### Step D: sort by the mapped key (what the CUB/HIPCUB call does)

The code does (conceptually):

- **input keys**: `blocksID_mapped`
- **output keys**: `blocksID_mapped_sorted`
- **input values**: `blocksLID_unsorted`
- **output values**: `blocksLID`

After sorting ascending by key:

- `blocksID_mapped_sorted = [1, 2, 4, 6, 7]`
- `blocksLID           = [LID(A), LID(C), LID(B), LID(D), LID(E)]`

So the sorted list is grouped by the mapped structure, not by the original sparse order.

### Step E: what `construct_columns_kernel` sees

In the kernel:

- `DX = gridLength[dimension] = D1 = 3`
- for each sorted mapped key \(m\):
  - `column_id = m / DX`
  - `dimension_id = m % DX`

Compute:

- $m=1$: `(column_id, dimension_id) = (0,1)`
- $m=2$: `(0,2)`
- $m=4$: `(1,1)`
- $m=6$: `(2,0)`
- $m=7$: `(2,1)`
Now “columns” are exactly the contiguous runs where:

- `column_id` stays the same, and
- `dimension_id` increases by 1 each step

So we get:

- `column_id=0`: `dimension_id 1→2` (length 2)
- `column_id=1`: `dimension_id 1` (length 1)
- `column_id=2`: `dimension_id 0→1` (length 2)

This is the whole point of “mapped + sort”: it turns “walk along y within a fixed (x,z)” into “walk along consecutive
integers in the sorted key space”.

---

## How “sorting by the mapped key” happens in the code

`construct_columns_kernel` does **no sorting itself**. The sorting happens earlier inside `sortBlocklistByDimension(...)`
using CUB (CUDA) or hipCUB (HIP):

- First, a mapping kernel writes:
  - keys: `blocksID_mapped[LID] = mapped(GID(LID))`
  - values: `blocksLID_unsorted[LID] = LID`
- Then radix sort is called as **SortPairs**:
  - **keys in**: `blocksID_mapped`
  - **keys out**: `blocksID_mapped_sorted`
  - **values in**: `blocksLID_unsorted`
  - **values out**: `blocksLID`

This is why later stages can:

- scan `blocksID_mapped_sorted` to discover column/set boundaries (purely from sorted keys), and
- use `blocksLID` / `blocksGID` (built from `blocksLID`) to access the actual per-block data in the new, grouped order.

---

## A smaller 2D example (same idea, less mental load)

This section is purely to make “mapped key + sorting” intuitive. In 2D there are only $(x,y)$, so:

- choosing `dimension = 0` means “columns run along x” (fixed y)
- choosing `dimension = 1` means “columns run along y” (fixed x)

Assume a 2D block grid with:

- `D0 = 4` blocks in x (x = 0..3)
- `D1 = 3` blocks in y (y = 0..2)

Assume the sparse active blocks are:

- A: $(x,y)=(0,1)$
- B: $(1,1)$
- C: $(3,1)$   (note the gap at x=2)
- D: $(2,0)$

### Goal: build “columns along x” (so x is fastest-varying)

Set `dimension = 0`, so `DX = gridLength[dimension] = D0 = 4`.

Choose a mapped key where **x is the remainder**:

$$
mapped = x + D0\cdot y
$$

Compute the mapped key for each active block:

- A $(0,1)$: $mapped = 0 + 4\cdot 1 = 4$
- B $(1,1)$: $mapped = 1 + 4\cdot 1 = 5$
- C $(3,1)$: $mapped = 3 + 4\cdot 1 = 7$
- D $(2,0)$: $mapped = 2 + 4\cdot 0 = 2$

### What “sorting by the mapped key” does

Say the sparse storage (LID order) happens to list blocks as: `[C, A, D, B]`.
Then the **unsorted keys** in that same order are: `[7, 4, 2, 5]`.

Radix sort (SortPairs) sorts by key ascending and permutes the carried values the same way:

| key (mapped) | block |
|-------------:|:------|
| 2            | D     |
| 4            | A     |
| 5            | B     |
| 7            | C     |

So after sorting:

- `blocksID_mapped_sorted = [2, 4, 5, 7]`
- `blocksLID` becomes `[LID(D), LID(A), LID(B), LID(C)]` (the same reordering)

### Now see why `construct_columns_kernel` can detect columns by “+1”

With `DX = 4`, the kernel computes for each sorted key `m`:

- `column_id = m / DX`  (here: effectively y)
- `dimension_id = m % DX` (here: x)

Compute:

- $m=2$: `(column_id, dimension_id) = (0,2)`  -> y=0, x=2
- $m=4$: `(1,0)` -> y=1, x=0
- $m=5$: `(1,1)` -> y=1, x=1
- $m=7$: `(1,3)` -> y=1, x=3

Now focus on `column_id = 1` (i.e. y=1). The `dimension_id` sequence is:

```text
x = 0, 1, 3
```

That contains a **gap** (missing x=2), so the kernel will split it into two column segments:

- segment 1: x=0→1 (contiguous, +1)
- segment 2: x=3 (after the gap)

That is exactly what “mapped + sort” buys you:

- blocks with the same `column_id` (same y) are adjacent in the sorted list, and
- within that group, contiguous x positions appear as `dimension_id` increasing by 1, so a gap is easy to detect.


## Important assumptions / invariants to keep in mind

These are relevant when you interpret the `[]` markers and when you reason about correctness/performance.

- **Single-block launch.**
  - The kernel assumes `gridDim` product is 1 and uses shared state + `__syncthreads()`.
- **`blocksID_mapped_sorted` is globally sorted and grouped by `column_id`.**
  - This relies on the prior radix sort and on the mapping definition.
- **`gpu_columnNBlocks[column_id]` must match the grouping in `blocksID_mapped_sorted`.**
  - `construct_columns_kernel` uses `blocks_in_columnset` to index into `blocksID_mapped_sorted[i + blocks_in_columnset - 1]`.
  - If counts are wrong (or not reset correctly), the kernel can read past the end or mis-segment columns.
- **Warp-size expectation in the ballot path.**
  - The code uses `__ffs(ballot_result)` and a literal `+32` when no bit is set.
  - That implicitly assumes the ballot width / active mask corresponds to 32 lanes (typical CUDA warp).
  - If `GPUTHREADS` or the platform’s warp/wavefront semantics differ, this needs careful review.
- **`columnData` must be empty before pushing.**
  - The kernel does not clear `columnData`; it only appends initial zeros and then more offsets/lengths.
  - Clearing currently happens earlier (in the surrounding workflow) and must be correct.

---

## How to use the outputs later (why this is done)

Downstream acceleration kernels often want to iterate “along a column” (contiguous blocks along the chosen dimension),
because many operations become 1D-like once you fix the two orthogonal coordinates.

This kernel turns a flat sorted list of blocks into a structure that supports:

- iterating over each contiguous column segment: `(start = columnBlockOffsets[c], len = columnNumBlocks[c])`
- iterating over each column set (all segments for a `column_id`): from `setColumnOffsets[s]` for `setNumColumns[s]` columns

So later kernels can quickly jump to “all blocks in this column/columnset” without re-discovering boundaries from scratch.

---

## Relationship to the existing “[] marker” report

If you are specifically analyzing the `[Warp Divergence]` annotations, see:

- `reports/warp-divergence/warp-divergence-construct_columns_kernel.md`

That note classifies which branches are actually lane-divergent vs warp-uniform vs intentional leader-thread serialization.

