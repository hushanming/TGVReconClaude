#pragma once
#include "types.hpp"
#include "depth_map.hpp"
#include <string>
#include <vector>

// Binary cube file layout:
//   [0..3]   magic: 'T','G','V','1'
//   [4..7]   version: uint32 = 1
//   [8..11]  r_root: float
//   [12..15] cube_count: uint32
//   [16..]   cube_count * sizeof(OctreeCube) bytes, sorted by MortonCode

struct CubeFileHeader {
    char magic[4] = {'T', 'G', 'V', '1'};
    uint32_t version = 1;
    float r_root = 0.0f;
    uint32_t cube_count = 0;
};

// Build sorted OctreeCube list from a depth map and write to `out_path`.
// `scene_origin`: world-space corner of scene bounding box [origin, origin+2*r_root]^3
// `r_root`: half-size of root octree cube (scene AABB side = 2*r_root)
// Pixels with r_x == 0 (no valid neighbors) are skipped.
void build_cubes_from_depthmap(
    const DepthMap& dm,
    const Vec3f& scene_origin,
    float r_root,
    const std::string& out_path);

// Load cubes back from a cube file (for testing / downstream stages).
std::vector<OctreeCube> load_cubes(const std::string& path, float& r_root_out);

// Load a contiguous slice [first_idx, last_idx] (inclusive) of cubes without
// reading the whole file.  r_root_out receives the scene scale from the header.
std::vector<OctreeCube> load_cubes_range(const std::string& path,
                                          uint32_t first_idx,
                                          uint32_t last_idx,
                                          float& r_root_out);

// Compute target octree depth for a sample radius r_x.
// Returns d such that 0.75*r_x <= r_root/2^d < 1.5*r_x.
// Clamps to [0, max_depth].
int cube_depth_for_radius(float r_x, float r_root, int max_depth = 21);
