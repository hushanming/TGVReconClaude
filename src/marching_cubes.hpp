#pragma once
#include "types.hpp"
#include "octree_builder.hpp"
#include "tgv.hpp"
#include <string>
#include <vector>

// A triangle in the output mesh.
struct Triangle {
    Vec3f v[3];
};

// Extract the u=0 iso-surface from the TGV result.
// For each octree cube, checks sign changes of u across the 6 faces;
// triangulates using the standard 15-case Marching Cubes lookup.
// Neighbouring cubes at different octree depths are handled by using
// the u value of the coarser cube's face as the boundary value.
//
// cubes:  sorted balanced OctreeCube array.
// state:  TGV state (same order as cubes).
// origin: world-space corner of scene AABB.
// r_root: half-size of root cube.
// has_data[i] = true if cube i has at least one histogram observation.
// Triangles are only emitted between cubes where both sides have data,
// preventing boundary artifacts at the observed/unobserved interface.
std::vector<Triangle> extract_surface(
    const std::vector<OctreeCube>& cubes,
    const std::vector<TGVCube>& state,
    const Vec3f& origin,
    float r_root,
    const std::vector<bool>& has_data = {});

// Write triangles to a Wavefront .obj file.
void write_obj(const std::string& path, const std::vector<Triangle>& tris);
