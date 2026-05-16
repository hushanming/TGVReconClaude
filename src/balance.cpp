#include "balance.hpp"
#include "octree_builder.hpp"
#include <algorithm>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------- helpers ----------------------------------------------------------

static uint64_t code64(const MortonCode& m) {
    return (static_cast<uint64_t>(m.hi) << 32) | m.lo;
}

static MortonCode from64(uint64_t v) {
    return {static_cast<uint32_t>(v >> 32), static_cast<uint32_t>(v)};
}

// Ancestor of a fine-grid code at target_depth (clears lower 3*(21-target_depth) bits).
static uint64_t ancestor64(uint64_t v, int target_depth) {
    unsigned shift = 3u * (21u - static_cast<unsigned>(target_depth));
    if (shift >= 64u) return 0;
    return v & ~((1ULL << shift) - 1ULL);
}

MortonCode parent_code(const MortonCode& code, int current_depth) {
    return from64(ancestor64(code64(code), current_depth - 1));
}

MortonCode face_neighbor(const MortonCode& code, int face, int depth) {
    uint32_t ix, iy, iz;
    morton_decode(code, ix, iy, iz);

    uint32_t step  = 1u << (21 - depth);
    uint32_t bound = 1u << 21;

    switch (face) {
        case 0: if (ix + step >= bound) return {UINT32_MAX, UINT32_MAX}; ix += step; break;
        case 1: if (ix < step)          return {UINT32_MAX, UINT32_MAX}; ix -= step; break;
        case 2: if (iy + step >= bound) return {UINT32_MAX, UINT32_MAX}; iy += step; break;
        case 3: if (iy < step)          return {UINT32_MAX, UINT32_MAX}; iy -= step; break;
        case 4: if (iz + step >= bound) return {UINT32_MAX, UINT32_MAX}; iz += step; break;
        case 5: if (iz < step)          return {UINT32_MAX, UINT32_MAX}; iz -= step; break;
        default: return {UINT32_MAX, UINT32_MAX};
    }
    return morton_encode(ix, iy, iz);
}

// ---------- balance ----------------------------------------------------------

// Encode (depth, fine-grid code) as a single 64-bit key.
// depth occupies bits 60-63 (max depth 21 fits in 5 bits), code64 in bits 0-59.
// We use a separate unordered_map<uint64_t, float> for r_c storage.
static uint64_t dc_key(int depth, uint64_t c) {
    return (static_cast<uint64_t>(depth) << 60) | (c & 0x0fffffffffffffffull);
}

void balance_octree(const std::string& in_path, const std::string& out_path) {
    float r_root;
    std::vector<OctreeCube> cubes = load_cubes(in_path, r_root);

    // Fast lookup: dc_key → (r_c_sum, count).
    // Average r_c when multiple cubes map to the same (depth, code) cell.
    std::unordered_map<uint64_t, float> rc_map;
    std::unordered_map<uint64_t, uint32_t> rc_cnt;
    rc_map.reserve(cubes.size() * 2);
    for (auto& c : cubes) {
        uint64_t k = dc_key(c.depth, code64(c.code));
        rc_map[k]  += c.r_c;
        rc_cnt[k]  += 1;
    }
    for (auto& [k, v] : rc_map)
        v /= static_cast<float>(rc_cnt[k]);

    // Per-depth sets for fast ancestor checks (depth 0..21).
    // We store fine-grid code64 values per depth level.
    std::vector<std::unordered_set<uint64_t>> by_depth(22);
    for (auto& c : cubes)
        by_depth[c.depth].insert(code64(c.code));

    // BFS queue: (depth, code64) pairs that need their neighbors checked.
    // Seed with all original cubes.
    std::queue<std::pair<int, uint64_t>> q;
    for (auto& c : cubes)
        q.push({c.depth, code64(c.code)});

    while (!q.empty()) {
        auto [d, cv] = q.front();
        q.pop();

        if (d < 2) continue;

        MortonCode code = from64(cv);
        for (int f = 0; f < 6; ++f) {
            MortonCode nb = face_neighbor(code, f, d);
            if (nb.hi == UINT32_MAX) continue;

            uint64_t nbv = code64(nb);

            // Same-level neighbor exists → OK
            if (by_depth[d].count(nbv)) continue;

            // Parent of nb at d-1 exists → OK (diff=1)
            uint64_t nb_par_v = ancestor64(nbv, d - 1);
            if (by_depth[d - 1].count(nb_par_v)) continue;

            // Check for coarser ancestor (diff ≥ 2) → violation: insert bridge at d-1
            bool violation = false;
            for (int d2 = d - 2; d2 >= 0; --d2) {
                uint64_t anc = ancestor64(nbv, d2);
                if (by_depth[d2].count(anc)) { violation = true; break; }
            }
            if (!violation) continue;

            // Insert bridge cube at (d-1, nb_par_v) if not already present.
            if (rc_map.emplace(dc_key(d - 1, nb_par_v),
                               r_root / static_cast<float>(1u << (d - 1))).second) {
                by_depth[d - 1].insert(nb_par_v);
                q.push({d - 1, nb_par_v});
            }
        }
    }

    // Reconstruct sorted cube list.
    std::vector<OctreeCube> out_cubes;
    out_cubes.reserve(rc_map.size());
    for (auto& [k, rc] : rc_map) {
        int dep = static_cast<int>(k >> 60);
        uint64_t cv = k & 0x0fffffffffffffffull;
        OctreeCube c;
        c.depth = dep;
        c.code  = from64(cv);
        c.r_c   = rc;
        out_cubes.push_back(c);
    }
    std::sort(out_cubes.begin(), out_cubes.end(),
              [](const OctreeCube& a, const OctreeCube& b) { return a.code < b.code; });

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + out_path);
    CubeFileHeader hdr;
    hdr.r_root     = r_root;
    hdr.cube_count = static_cast<uint32_t>(out_cubes.size());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!out_cubes.empty())
        out.write(reinterpret_cast<const char*>(out_cubes.data()),
                  out_cubes.size() * sizeof(OctreeCube));
}
