#pragma once
#include "types.hpp"
#include "octree_builder.hpp"
#include "tgv.hpp"
#include <string>
#include <vector>

// Indexed triangle mesh with deduplicated vertices.
struct Mesh {
    std::vector<Vec3f>    verts;         // unique vertex positions
    std::vector<uint32_t> faces;         // 3 indices per triangle, CCW winding
    std::vector<Vec3f>    normals;       // per-vertex averaged face normals (optional)

    bool   empty()     const { return faces.empty(); }
    size_t tri_count() const { return faces.size() / 3; }
};

// Extract the u=0 iso-surface from TGV state as an indexed mesh.
// Vertices that lie at the same quantised position are merged, giving a
// manifold mesh for each connected sign-change surface.
//
// cubes    : balanced OctreeCube array (Morton-sorted).
// state    : TGV state, same order.
// origin   : world-space lower-corner of scene AABB.
// r_root   : half-size of root cube.
// has_data : if non-empty, triangles are only emitted when both adjacent
//            cubes have at least one histogram vote (avoids spurious patches).
Mesh extract_surface(
    const std::vector<OctreeCube>& cubes,
    const std::vector<TGVCube>&    state,
    const Vec3f&                   origin,
    float                          r_root,
    const std::vector<bool>&       has_data = {});

// Streaming out-of-core surface extraction.
// Reads cubes / state / histograms from disk a treetop leaf at a time, applies
// an ownership rule on cube-pair iteration so each face pair is emitted exactly
// once, and writes the final closed mesh directly to a Wavefront .obj file.
//
// Peak RAM is bounded by a single leaf's data + a vertex dedup map sized
// proportional to the final surface area (≪ total cube count).
//
// has_data filtering is implicit: faces are emitted only when both adjacent
// cubes have at least one histogram vote.
void extract_surface_streaming(
    const std::string& balanced_cube_path,
    const std::string& tgv_state_path,
    const std::string& treetop_path,
    const std::string& hist_path,
    const Vec3f&       origin,
    float              r_root,
    const std::string& out_obj_path);

// Compute per-vertex normals from face normals and store in mesh.normals.
// Normals are averaged over adjacent faces and normalised.
void compute_normals(Mesh& mesh);

// Write mesh to Wavefront .obj (vertices + faces; normals if present).
void write_obj(const std::string& path, const Mesh& mesh);

// Triangle (kept for legacy use in small unit tests).
struct Triangle { Vec3f v[3]; };
