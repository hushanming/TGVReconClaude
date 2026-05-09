#pragma once
#include "types.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
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

// Run TGV minimization on the balanced octree.
// cubes:     sorted balanced OctreeCube array (fine-grid Morton codes).
// hists:     8-bin histograms (one per cube, same order).
// origin:    world-space corner of scene AABB.
// r_root:    half-size of root cube.
// Returns per-cube TGV state (u values are the output indicator field).
std::vector<TGVCube> tgv_minimize(
    const std::vector<OctreeCube>& cubes,
    const std::vector<CubeHistogram>& hists,
    const Vec3f& origin,
    float r_root,
    const TGVParams& params = TGVParams{});
