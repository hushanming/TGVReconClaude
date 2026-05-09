This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Implementation of the ICCV 2021 paper: **"Out-of-Core Surface Reconstruction via Global TGV Minimization"** (Nikolai Poliarnyi, Agisoft LLC). The paper is in the repo as a MinerU-converted markdown.

Goal: reconstruct 3D surfaces from depth maps and terrestrial LIDAR scans with strict ≤16 GB peak RAM, regardless of dataset size, via GPU-accelerated out-of-core TGV minimization on a 2:1 balanced adaptive octree.

## Intended Tech Stack

No build system exists yet. Dependencies visible in `.claude/settings.local.json`:
- **Eigen3** — linear algebra
- **libomp** — OpenMP parallelism
- **Intel TBB** — threading
- **OpenCL** — GPU acceleration (histogram computation + primal-dual iterations)

## Algorithm Architecture

Sequential out-of-core pipeline. Each stage processes data in Morton-code-sorted chunks; all merges are out-of-core k-way merge sorts.

### Stage 1: Distance Field Generation
Each depth map pixel / LIDAR point → sample with center `p_x` and radius `r_x` (half-distance to neighboring pixel in 3D). Encoded as distance field `f_i`: value +1 on ray between camera and surface, fades to -1 behind surface. Parameters: `δ_x = 6·r_x` (near-surface width), `η_x = 18·r_x` (occluded region width). LIDAR treated as noise-free depth maps of 360° cameras.

### Stage 2: Linear Octree Build
Per sample, spawn octree cube at depth `d` satisfying `0.75·r_x ≤ r_root/2^d < 1.5·r_x`. Cubes encoded as **96-bit 3D Morton codes**. One file per depth map → k-way merge sort into single sorted linear octree. Per-cube density `r_c = mean(r_x for x in cube)` stored for LOD selection.

### Stage 3: 2:1 Octree Balancing
Balance constraint: each cube has ≤4 neighbors per face. Process sorted linear octree in chunks → balance each independently → merge balanced chunks (k-way merge sort). Relies on Morton code ordering throughout.

### Stage 4: Treetop Indexing
Build minimum subtree (treetop) where each leaf covers < `N_cubesPerTreetopLeaf = 2²⁴` descendants. Each leaf stores `[first_idx, last_idx]` into the balanced octree (consecutive due to Z-curve) — enables IO-friendly sequential reads. Limits to ~thousands of leaves even on billion-point datasets.

### Stage 5: Histogram Computation (GPU/OpenCL)
Per treetop leaf: load leaf cubes + relevant depth maps (frustum intersection test). For each cube-depthmap pair, project cube center into **depth map mipmap pyramid** at LOD matching `voxel.radius`, accumulate into **8-bin histogram** (following [Zach 2008]). LIDAR votes weighted ×5. Memory bound: `N_cubesPerTreetopLeaf` cubes + 1 depth map at a time.

Algorithm 1 (histogram bin computation):
```
mipmap_level, pixel = depthmap_pyramid.project(voxel.center, voxel.radius)
a = depthmap.depth(pixel) - distance_to(voxel.center)
if a < -η_x: skip  # occluded
a = clamp(a / δ_x, -1, 1)
bin = floor((a + 1.0) / 2.0 * 8.0)
voxel.histogram[bin] += vote  # vote=1 for photos, 5 for LIDAR
```

### Stage 6: TGV Minimization (GPU/OpenCL)
**Functional:** `min_{u,v} { α₁|∇u−v| + α₀|ε(v)| + Σᵢ|u−fᵢ| }` where `ε(v) = (∇v + ∇vᵀ)/2`.

Coarse-to-fine across octree levels (200 primal-dual iterations per level). Per treetop leaf: load set A (leaf cubes) + set B (border neighbors). Run iterations updating A only; B values frozen to parent-level indicator (eliminates seams at leaf boundaries). On coarsest levels, process multiple leaves together (few cubes per leaf).

**Per-cube data:** Morton code (96-bit), `r_c`, 8-bin histogram, primal vars `u` (indicator scalar), `v` (vector field), dual vars `p`, `q`.

### Stage 7: Marching Cubes + Decimation
Per leaf: extract iso-surface at `u=0`. For each cube, find sign changes in `u` across faces → triangulate via dynamic programming minimizing surface area. Follow with **QSlim decimation** — constraint: never collapse border edges (edges on treetop-leaf cube faces) to guarantee seamless joins between leaves.

## Key Design Invariants

- Morton Z-curve ordering is the backbone of all out-of-core operations
- `r_c` density per cube makes mipmap LOD selection invariant to bounding-box size changes
- Coarse-to-fine minimization + frozen border values = no seams between treetop leaves
- `N_cubesPerTreetopLeaf = 2²⁴` chosen to bound each stage to ≤16 GB RAM

硬性要求：
1. 最终网格必须是封闭实体，没有边界边。
2. 每条边恰好被两个三角形共享，没有悬垂面。
3. 没有重复顶点、重复面、零面积三角形。
4. 所有法线向外一致。
