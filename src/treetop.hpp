#pragma once
#include "types.hpp"
#include <string>
#include <vector>

// One leaf of the treetop index.
// Corresponds to a contiguous range of cubes in the balanced octree file.
struct TreetopLeaf {
    MortonCode code;      // fine-grid Morton code of this node (lower free-bits are zero)
    int32_t    depth;     // depth of this node in the treetop
    uint32_t   first_idx; // first cube index (inclusive) in balanced octree
    uint32_t   last_idx;  // last cube index (inclusive)
};

// File layout:
//   [0..3]   magic 'T','G','V','T'
//   [4..7]   version uint32 = 1
//   [8..11]  r_root float
//   [12..15] leaf_count uint32
//   [16..]   leaf_count * sizeof(TreetopLeaf) bytes

static constexpr uint32_t N_CUBES_PER_TREETOP_LEAF = 1u << 24;  // 16M

// Build treetop index from a sorted balanced OctreeCube file.
void build_treetop(const std::string& in_cube_path, const std::string& out_treetop_path);

// Load treetop leaves. Sets r_root_out to the scene half-size.
std::vector<TreetopLeaf> load_treetop(const std::string& path, float& r_root_out);
