#pragma once
#include "types.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
#include "treetop.hpp"
#include <string>
#include <vector>

// Per-cube TGV primal and dual variables.
// Primal:   u (scalar indicator), v (vector field)
// Dual:     p (3-vec for |∇u−v|), q (6-vec for |ε(v)|, Voigt order xx yy zz xy xz yz)
struct TGVCube {
    float u    = 0.0f;
    float v[3] = {};
    float p[3] = {};
    float q[6] = {};   // Voigt: xx, yy, zz, xy, xz, yz
};

// File layout: magic 'TGVS', version 1, count uint32, TGVCube[]
void save_tgv(const std::string& path, const std::vector<TGVCube>& state);
std::vector<TGVCube> load_tgv(const std::string& path);

// TGV minimization parameters.
struct TGVParams {
    float alpha0 = 0.02f;   // weight on |ε(v)|  (smoothness of v)
    float alpha1 = 0.04f;   // weight on |∇u−v|  (coupling u to v)
    int   iters  = 200;     // primal-dual iterations per level
};

// Run TGV minimization on the balanced octree (in-memory).
std::vector<TGVCube> tgv_minimize(
    const std::vector<OctreeCube>& cubes,
    const std::vector<CubeHistogram>& hists,
    const Vec3f& origin,
    float r_root,
    const TGVParams& params = TGVParams{});

// Out-of-core TGV minimization: one treetop leaf at a time per depth level.
// Border neighbors (same depth, outside the current leaf) are frozen to their
// pre-level snapshot values — this eliminates seams at leaf boundaries.
// When leaves covers all cubes (single leaf), results are identical to tgv_minimize.
std::vector<TGVCube> tgv_minimize_oc(
    const std::vector<OctreeCube>& cubes,
    const std::vector<CubeHistogram>& hists,
    const std::vector<TreetopLeaf>& leaves,
    const Vec3f& origin,
    float r_root,
    const TGVParams& params = TGVParams{});

// Streaming out-of-core TGV minimization. Inputs and the resulting TGV state
// live entirely on disk; peak RAM is bounded by a single treetop leaf
// (≈ N_CUBES_PER_TREETOP_LEAF * (sizeof(OctreeCube) + sizeof(CubeHistogram)
//    + sizeof(TGVCube) + 16-byte bars) + border data).
//
// Produces `out_state_path` in the same format used by save_tgv / load_tgv,
// so Stage 7 can consume it unchanged.
void tgv_minimize_streaming(
    const std::string& balanced_cube_path,
    const std::string& hist_path,
    const std::string& treetop_path,
    const Vec3f&       origin,
    float              r_root,
    const std::string& out_state_path,
    const TGVParams&   params = TGVParams{});
