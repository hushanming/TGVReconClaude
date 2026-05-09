#include "treetop.hpp"
#include "octree_builder.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

struct TreetopFileHeader {
    char     magic[4] = {'T', 'G', 'V', 'T'};
    uint32_t version  = 1;
    float    r_root   = 0.0f;
    uint32_t leaf_count = 0;
};

// ---------- build ------------------------------------------------------------

// The sorted cube array uses fine-grid Morton codes.  All cubes in a depth-d
// octree node occupy a contiguous Z-curve interval, enabling binary search.
// Node at depth d with fine-grid code `node_code` spans Z-curve values
//   [node_code,  node_code | ((1<<(3*(21-d)))-1)]
// Octant q (bits dx|dy<<1|dz<<2) of depth d occupies the sub-interval at d+1:
//   offset = q << (3*(20-d))     (the three new bits at depth d+1)

static void build_recursive(
    const std::vector<OctreeCube>& cubes,
    uint32_t first, uint32_t last,   // half-open range [first, last)
    uint64_t node_code64,            // 64-bit interleaved code (lower bits zero)
    int      depth,
    std::vector<TreetopLeaf>& leaves)
{
    uint32_t count = last - first;
    if (count == 0) return;

    if (count <= N_CUBES_PER_TREETOP_LEAF || depth >= 21) {
        TreetopLeaf leaf;
        leaf.code      = {static_cast<uint32_t>(node_code64 >> 32),
                          static_cast<uint32_t>(node_code64)};
        leaf.depth     = depth;
        leaf.first_idx = first;
        leaf.last_idx  = last - 1;
        leaves.push_back(leaf);
        return;
    }

    // Split into 8 children at depth+1.
    // Child q occupies Z-curve bits shifted by 3*(20-depth).
    unsigned child_shift = (depth <= 20u) ? 3u * (20u - static_cast<unsigned>(depth)) : 0u;

    uint32_t child_first = first;
    for (int q = 0; q < 8; ++q) {
        uint64_t child_code64 = node_code64 | ((uint64_t)q << child_shift);
        // Upper bound (inclusive) for this child's Z-curve range.
        uint64_t child_max64  = child_code64 | ((child_shift > 0) ? ((1ULL << child_shift) - 1ULL) : 0ULL);

        MortonCode child_max = {static_cast<uint32_t>(child_max64 >> 32),
                                static_cast<uint32_t>(child_max64)};

        // Binary search: first index with code > child_max.
        uint32_t lo = child_first, hi = last;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (!(child_max < cubes[mid].code)) lo = mid + 1;
            else                                hi = mid;
        }
        uint32_t child_last = lo;

        build_recursive(cubes, child_first, child_last, child_code64, depth + 1, leaves);
        child_first = child_last;
        if (child_first >= last) break;
    }
}

void build_treetop(const std::string& in_cube_path, const std::string& out_treetop_path) {
    float r_root;
    std::vector<OctreeCube> cubes = load_cubes(in_cube_path, r_root);

    std::vector<TreetopLeaf> leaves;
    build_recursive(cubes, 0, static_cast<uint32_t>(cubes.size()), 0ULL, 0, leaves);

    std::ofstream out(out_treetop_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + out_treetop_path);

    TreetopFileHeader hdr;
    hdr.r_root     = r_root;
    hdr.leaf_count = static_cast<uint32_t>(leaves.size());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!leaves.empty())
        out.write(reinterpret_cast<const char*>(leaves.data()),
                  leaves.size() * sizeof(TreetopLeaf));
}

// ---------- load -------------------------------------------------------------

std::vector<TreetopLeaf> load_treetop(const std::string& path, float& r_root_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open treetop: " + path);

    TreetopFileHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='T')
        throw std::runtime_error("Bad treetop magic: " + path);

    r_root_out = hdr.r_root;
    std::vector<TreetopLeaf> leaves(hdr.leaf_count);
    if (hdr.leaf_count > 0)
        in.read(reinterpret_cast<char*>(leaves.data()),
                hdr.leaf_count * sizeof(TreetopLeaf));
    return leaves;
}
