# Vlasiator velocity-space primer (data model + mental model)

This document explains how Vlasiator represents and evolves the Vlasov distribution function
$f_s(\mathbf{x},\mathbf{v},t)$ in the codebase, with emphasis on:

- the relationship between **spatial cells** and **velocity space**
- the difference between **velocity mesh parameters** vs **sparse per-cell velocity blocks**
- what **blocks** and **bins/cells inside blocks** are
- what `WID`, `GID`, `LID`, and `gridLength` mean
- where “$f$ is computed” (initialization / boundary conditions) and where it is stored

The goal is to give you a stable mental model you can reuse while reading kernels like
`construct_columns_kernel` or remapping/acceleration code.

---

## Big picture: what is being solved?

Vlasiator is a Vlasov code: for each species/population $s$, it evolves a distribution function
$f_s(\mathbf{x},\mathbf{v},t)$, i.e. “how much phase-space density exists at velocity $\mathbf{v}$ inside spatial
cell $\mathbf{x}$”.

This is not a “particle has velocity” model. Instead:

- **Spatial domain** is discretized into **spatial cells** (grid in $\mathbf{x}$)
- For each spatial cell and each species, the state includes:
  - **bulk moments** (density, bulk velocity, pressure, …)
  - a **3D velocity-space distribution** sampled on a 3D velocity grid

---

## The three nested grids (the source of most confusion)

### 1) Spatial grid (physical domain)

Each `SpatialCell` corresponds to a small region in physical space. It stores its corner and size,
e.g. `CellParams::XCRD/YCRD/ZCRD` and `CellParams::DX/DY/DZ`.

See enum documentation in `common.h`:

- `CellParams::XCRD/YCRD/ZCRD` = bottom-left corner (conceptually)
- `CellParams::DX/DY/DZ` = cell size

### 2) Velocity *mesh* (velocity-domain geometry, per species)

Each species has a configured velocity-domain box $[v_{\min},v_{\max})$ in each direction, split into a grid.

This geometry is stored in `vmesh::MeshParameters`:

- `meshLimits[6]`: vx_min, vx_max, vy_min, vy_max, vz_min, vz_max
- `gridLength[3]`: **number of velocity blocks** in each direction
- `blockLength[3]`: **number of velocity bins per block** in each direction (typically `WID`)
- derived: `blockSize[3]`, `cellSize[3]`, etc.

### 3) Velocity blocks and velocity bins (the discretization of $f$, per spatial cell + species)

The actual discretized $f$ is stored as:

- a **sparse set of velocity blocks** for each (spatial cell, species)
- each velocity block contains **`WID × WID × WID` velocity bins**
- each bin holds one stored scalar $f$ value representing $f$ over a small velocity-volume element

Sparse means: most velocity space is usually empty/negligible in a given spatial cell, so Vlasiator only stores
blocks that matter.

---

## Species/populations: the key glue between spatial cells and velocity space

Every `SpatialCell` owns a `std::vector<Population> populations` (one entry per configured particle population).

Each `Population` owns:

- bulk moments (`RHO`, `V[3]`, `P[3]`, …)
- a per-cell `vmesh::VelocityMesh* vmesh` (which velocity blocks exist; mapping IDs)
- a per-cell `vmesh::VelocityBlockContainer* blockContainer` (the actual $f$ values + per-block parameters)

This is the core “link” between x-space and v-space in the code.

---

## Key identifiers: `WID`, `GID`, `LID`

### `WID`

`WID` is the number of velocity bins per axis *inside one velocity block*.
The default is historically 4, so one block contains `WID^3 = 64` bins.

`cellIndex(i,j,k)` flattens a bin index within the block.

### `GID` (GlobalID)

A velocity **block** has a 3D block index $(i_\mathrm{block},j_\mathrm{block},k_\mathrm{block})$ inside the full
velocity-block grid. Vlasiator flattens that to a `vmesh::GlobalID` (dense identity within the configured velocity box).

You should think: **GID identifies “which block in the full velocity grid?”** even if that block is not stored.

### `LID` (LocalID)

`vmesh::LocalID` is the index of a velocity block **in the sparse list of blocks stored in one (spatial cell, species)**.

You should think: **LID is “where is this block’s data in the compact arrays?”**

The mapping between `GID ↔ LID` is maintained by `Population::vmesh`.

---

## `gridLength`: what it is (and what it is not)

`gridLength[3]` is part of the **velocity mesh parameters**. It is:

- number of velocity blocks in vx direction
- number of velocity blocks in vy direction
- number of velocity blocks in vz direction

It is **not** the spatial grid resolution.

The maximum number of possible blocks for that species’ velocity mesh is:

$$
\text{max\_velocity\_blocks} = gridLength_x \cdot gridLength_y \cdot gridLength_z
$$

This is a geometric upper bound; the actual number of active blocks per cell is `vmesh->size()`, and is dynamic.

---

## Indices vs physical coordinates (very important)

Most of the time, the code operates on **indices** (which bin/block in memory), not on “velocity coordinates stored in bins”.

- **Pre-known (from mesh parameters/config):**
  - velocity-domain limits $v_{min}$ / $v_{\max}$
  - velocity cell size(s) $dv_x,dv_y,dv_z$ (stored/derived via `cellSize[]`)
  - block size(s) (stored/derived via `blockSize[]`)
  - `WID` (bins per block per axis)
- **Stored at runtime (per spatial cell + species):**
  - which blocks exist (sparse set): `GID ↔ LID` mapping in `VelocityMesh`
  - the distribution values: one scalar $f$ value per bin in `VelocityBlockContainer`

When you *need* a physical velocity coordinate for a given bin, you **derive** it from indices plus mesh parameters:

- **1D analogy:** bin index $n$ corresponds to the interval
  $[v_{\min}+n\,dv,\;v_{\min}+(n+1)\,dv)$, and a representative velocity is often
  $v_{\min}+(n+0.5)\,dv$.
- **3D:** the bin $(i,j,k)$ in a given block corresponds to the cube with edges defined by
  the block origin (VXCRD/VYCRD/VZCRD) and per-axis spacings (DVX/DVY/DVZ).

So: **bins do not “store velocities”**; they store $f$. The velocity coordinate is computed when needed.

---

## What is a “bin” in velocity space?

There is no explicit `struct Bin { ... }`.

A “velocity bin” is an **index into a flat array** of $f$ values:

- identify the block via `blockLID` (local block index, sparse)
- identify the bin inside that block via $(i,j,k)$
- flatten $(i,j,k)$ with `cellIndex(i,j,k)`
- access: `blockData[blockLID*WID3 + cellIndex(i,j,k)]`

In 1D analogy, a bin is indexed by a single integer `n` and represents an interval
$[v_{\min}+n\,dv,\;v_{\min}+(n+1)\,dv)$. The stored value represents $f$ over that interval; a representative
velocity is often the center $v_{\min}+(n+0.5)dv$. Importantly, the “range” is *derived* from $v_{\min}$ and $dv$;
the bin is fundamentally identified by its index.

In 3D, the bin corresponds to a small cube in $(v_x,v_y,v_z)$ with volume $dv_x\,dv_y\,dv_z$.

---

## $(v_x,v_y,v_z)$ vs the stored scalar $f$ (what is *actually* stored?)

This is the single most important distinction:

- **$(v_x,v_y,v_z)$ are coordinates** in velocity space (like “where you are on the velocity grid”).
- **$f$ is the value stored** at that location (like “how much phase-space density is there here?”).

### What a bin stores

A velocity bin stores **one scalar** (typically `Realf`) representing $f$ for the velocity-space volume element
associated with that bin.

It does *not* store $(v_x,v_y,v_z)$ numbers. Those are derived.

**Interpretation (plain language):** the stored scalar answers “how much of this species (phase-space density) is moving
with velocities near $(v_x,v_y,v_z)$ in this spatial cell”, where “near” means “within this bin’s small
velocity-volume element”.

### Tiny concrete example (numbers)

Assume (purely as an example) that for one species in one spatial cell:

- velocity-space cell sizes are $dv_x = dv_y = dv_z = 10^3\ \mathrm{m/s}$
- for a particular bin, the derived cell-centered velocity is $(v_x,v_y,v_z) = (4, -2, 7)\times 10^5\ \mathrm{m/s}$

Then the velocity-space volume element represented by that bin is:

$$
DV3 = dv_x\,dv_y\,dv_z = (10^3)^3 = 10^9\ (\mathrm{m/s})^3
$$

If the stored scalar in that bin is, say, $f = 2.0\times 10^{-12}$ (units depend on normalization),
then:

- **qualitatively:** there is “$f$ amount” of phase-space density moving near that velocity vector
- **when computing number density:** this bin contributes approximately $f\cdot DV3$ to $\int f\,d^3v$, i.e.

$$
\Delta n \approx f\cdot DV3 = 2.0\times 10^{-12}\times 10^9 = 2.0\times 10^{-3}
$$

(In real runs, the physical units/scaling are handled by the simulation’s unit system; the key point is the
pattern “stored $f$ times $DV3$ contributes to moments”.)

### What is the *physical* meaning of this “volume”?

Here “volume” does **not** mean a volume in real space. It means a **volume in velocity space**:

- A single bin corresponds to a small 3D box of velocities:
  - $v_x$ in a range of width $dv_x$
  - $v_y$ in a range of width $dv_y$
  - $v_z$ in a range of width $dv_z$
- That box has “volume” $DV3 = dv_x\,dv_y\,dv_z$ with units $(\mathrm{m/s})^3$ (or whatever velocity units the run uses).

Why does it matter?

- The distribution function is defined so that **“$f$ times a velocity-space volume”** tells you how much number density
  lives in that range of velocities.
- In continuous form, number density at position $\mathbf{x}$ is:
  - $n(\mathbf{x}) = \int f(\mathbf{x},\mathbf{v})\,d^3v$
- In discretized form, that integral becomes a sum over bins:
  - each bin contributes roughly “(value stored in bin) × (velocity-space volume of bin)”

So you can read $DV3$ as: **how wide the bin is in velocity space**, i.e. “how big a chunk of velocities this bin represents”.

### How $(v_x,v_y,v_z)$ are determined for a bin

For a given velocity block, the code stores per-block parameters:

- `VXCRD/VYCRD/VZCRD`: the block’s lower corner in velocity space
- `DVX/DVY/DVZ`: the per-bin spacing inside the block

Then for bin indices $(i,j,k)$ inside the block, a common convention is to use the **cell center**:

$$
v_x = VXCRD + (i+0.5)\,DVX,\quad
v_y = VYCRD + (j+0.5)\,DVY,\quad
v_z = VZCRD + (k+0.5)\,DVZ
$$

You can see this pattern explicitly in moment computation (reconstructing $(v_x,v_y,v_z)$ from block parameters and indices):

```cpp
// vlasovsolver/arch_moments.h (excerpt)
const Real VX = blockParamsZ[BlockParams::VXCRD] + (i+HALF)*blockParamsZ[BlockParams::DVX];
const Real VY = blockParamsZ[BlockParams::VYCRD] + (j+HALF)*blockParamsZ[BlockParams::DVY];
const Real VZ = blockParamsZ[BlockParams::VZCRD] + (k+HALF)*blockParamsZ[BlockParams::DVZ];
lsum[0] += avgs[cellIndex(i,j,k)] * DV3;
```

### How the scalar $f$ is computed for that bin

Initialization/boundary code does:

1) derive $(x,y,z)$ for the spatial cell (plus sizes $dx,dy,dz$)
2) derive $(v_x,v_y,v_z)$ for the velocity bin (plus sizes $dv_x,dv_y,dv_z$)
3) evaluate a formula returning a scalar (e.g. Maxwellian, or project-specific)
4) store that scalar into the bin slot

In `Project::setVelocityBlock(...)`, the scalar returned by `calcPhaseSpaceDensity(...)` is written directly to the bin:

```cpp
// projects/project.cpp (excerpt)
creal average =
  calcPhaseSpaceDensity(x,y,z, dx,dy,dz,
                        vxCell,vyCell,vzCell,
                        dvxCell,dvyCell,dvzCell, popID);
buffer[cellIndex(ic,jc,kc)] = average;
```

So: **the “computed scalar” is $f$**, and it is stored at the bin whose $(v_x,v_y,v_z)$ is implied by indices + mesh parameters.

**Quick note:** yes—during initialization/boundary setting, the *formula* computes $f$ (a velocity-space density). Physical
number density is then obtained by integrating/summing $f$ over velocity space (multiplying by $DV3$ per bin).

---

## Where does “$f$ is computed” happen?

## How $f$ connects to a bin (and how a bin value is computed/stored)

This is the concrete “bridge” between the math and the memory layout:

1) **A bin is a slot in an array** (addressed by indices).
2) **The code derives physical coordinates** \((x,y,z)\) and \((v_x,v_y,v_z)\) for that bin from:
   - spatial cell origin + size (`XCRD/YCRD/ZCRD`, `DX/DY/DZ`)
   - velocity block origin + velocity cell sizes (`VXCRD/VYCRD/VZCRD`, `DVX/DVY/DVZ`)
3) A project/boundary condition evaluates a **formula** that returns a scalar (the phase-space density value for that bin).
4) That scalar is written into the bin slot.

### The bin “slot”: `cellIndex(i,j,k)` inside one block

Within one velocity block, bin indices \((i,j,k)\) are flattened with `cellIndex(i,j,k)`. The bin is fundamentally
identified by indices; physical \((v_x,v_y,v_z)\) is derived when needed.

### Example: project initialization computes and stores $f$ per bin

The base `Project::setVelocityBlock(...)` (used during initialization) shows the full loop:

- It reads spatial-cell geometry (`x,y,z,dx,dy,dz`)
- It derives per-bin velocity coordinates from block origin plus `dv*index`
- It calls the (project-specific) formula `calcPhaseSpaceDensity(...)`
- It stores the result into the bin slot `buffer[cellIndex(ic,jc,kc)]`

```cpp
// projects/project.cpp (excerpt)
creal x  = cell->parameters[CellParams::XCRD];
creal y  = cell->parameters[CellParams::YCRD];
creal z  = cell->parameters[CellParams::ZCRD];
creal dx = cell->parameters[CellParams::DX];
creal dy = cell->parameters[CellParams::DY];
creal dz = cell->parameters[CellParams::DZ];

// block origin in velocity space (lower corner)
Real blockCoords[3];
cell->get_velocity_block_coordinates(popID,blockGID,&blockCoords[0]);
creal vxBlock = blockCoords[0];
creal vyBlock = blockCoords[1];
creal vzBlock = blockCoords[2];

// velocity bin sizes
creal dvxCell = cell->get_velocity_grid_cell_size(popID)[0];
creal dvyCell = cell->get_velocity_grid_cell_size(popID)[1];
creal dvzCell = cell->get_velocity_grid_cell_size(popID)[2];

for (uint kc=0; kc<WID_VZ; ++kc) {
  for (uint jc=0; jc<WID_VY; ++jc) {
    for (uint ic=0; ic<WID_VX; ++ic) {
      creal vxCell = vxBlock + ic*dvxCell;
      creal vyCell = vyBlock + jc*dvyCell;
      creal vzCell = vzBlock + kc*dvzCell;
      creal average =
        calcPhaseSpaceDensity(x,y,z, dx,dy,dz,
                              vxCell,vyCell,vzCell,
                              dvxCell,dvyCell,dvzCell, popID);
      buffer[cellIndex(ic,jc,kc)] = average;
    }
  }
}
```

The most important conceptual point: **the formula returns one scalar per bin**, and that scalar is what gets stored.

### Example: Maxwellian boundary condition fills bins from an explicit formula

Boundary conditions do the same “compute scalar → write to slot” workflow, but with a boundary-specific formula.
Example excerpt:

```cpp
// sysboundary/setmaxwellian.cpp (excerpt)
creal vxCell = vxBlock + (ic+0.5)*dvxCell - Vx;
creal vyCell = vyBlock + (jc+0.5)*dvyCell - Vy;
creal vzCell = vzBlock + (kc+0.5)*dvzCell - Vz;
Realf average = maxwellianDistribution(popID,rho,T,vxCell,vyCell,vzCell);
initBuffer[i*WID3+cellIndex(ic,jc,kc)] = average;
```

### How $f$ is later “used”: moments integrate $f$ over velocity-space volume

When converting $f$ into physical moments (density, momentum, …), the code treats each stored scalar as representing
a finite velocity-space volume element $DV3 = dv_x\,dv_y\,dv_z$, and reconstructs cell-centered velocities:

```cpp
// vlasovsolver/arch_moments.h (excerpt)
const Real DV3 = DVX*DVY*DVZ;
const Real VX = VXCRD + (i+0.5)*DVX;
// ...
lsum[0] += avgs[cellIndex(i,j,k)] * DV3;
lsum[1] += avgs[cellIndex(i,j,k)] * VX * DV3;
```

This is the “math connection”: $ \int f(\mathbf{v})\,d^3v \approx \sum f_{ijk}\,DV3 $.

### Initialization (fresh run)

At the start of a run, the project defines an initial condition $f_s(\mathbf{x},\mathbf{v},t=0)$.

The core pattern is:

1) choose a list of blocks to initialize (often a heuristic; base class can be expensive)
2) for each chosen block, loop over its bins and evaluate a formula (e.g. Maxwellian, project-specific)
3) write the computed values into a temporary buffer
4) add the blocks into the cell’s sparse `vmesh` + `blockContainer` (copy buffer into persistent storage)

In `projects/project.cpp`, the base `Project::setVelocityBlock(...)` shows the per-bin evaluation and storage:

- it reconstructs spatial cell geometry (`x,y,z,dx,dy,dz`)
- it reconstructs per-bin velocity coordinates from block origin and `dv*index`
- it calls `calcPhaseSpaceDensity(...)` to compute a value for the bin
- it writes `buffer[cellIndex(ic,jc,kc)] = average`

Then `Project::setVelocitySpace(...)` builds `initBuffer` for all blocks and calls `cell->add_velocity_blocks(...)`
to persist it.

### Boundary conditions

Boundary conditions can also compute $f$ values from a formula (for specific boundary types).
Example: `sysboundary/setmaxwellian.cpp` fills bins with `maxwellianDistribution(...)`.

### Time stepping (evolution)

After initialization, the solver updates the stored bin values each timestep. High level:

- **translation in space**: moves distribution between neighboring spatial cells based on velocity
- **acceleration in velocity space**: moves distribution between neighboring velocity bins based on acceleration

You’ll see this in kernels that read/write `blockContainer->getData(...)` and use `dv` and cell-centered velocity
\((index+0.5)\) forms.

---

## Sparse activation: how many blocks exist per (cell, species)?

Not static. For each spatial cell and each species:

- the current number of active blocks is `Population::vmesh->size()`
- blocks can be added/removed based on thresholding (`Species::sparseMinValue`) and advection/acceleration needs
- there is a geometric maximum `max_velocity_blocks`, but storing all of them would be dense and huge

---

## Minimal “read the code” map (where to look next)

- **Population + SpatialCell ownership**: `spatial_cell_cpu.hpp` and `spatial_cell_cpu.cpp`
- **Velocity mesh parameters (limits/gridLength/dv)**: `velocity_mesh_parameters.h/.cpp` and `object_wrapper.cpp`
- **Sparse mapping GID↔LID**: `velocity_mesh_gpu.h` (or `velocity_mesh_old.h`)
- **Storage of \(f\) values**: `velocity_block_container.h`
- **Initialization formula loop**: `projects/project.cpp` (`setVelocityBlock` / `setVelocitySpace`)
- **Moments (how \(f\) is interpreted physically)**: `vlasovsolver/arch_moments.h` (uses `DV3` and cell-centered velocities)
- **Acceleration “columns” concept** (GPU): `vlasovsolver/gpu_acc_sort_blocks.cpp` + `arch/gpu_base.hpp` (`ColumnOffsets`)

