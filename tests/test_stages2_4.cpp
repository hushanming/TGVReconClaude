#include "types.hpp"
#include "depth_map.hpp"
#include "octree_builder.hpp"
#include "merge.hpp"
#include "balance.hpp"
#include "treetop.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)

// ---- helpers ----------------------------------------------------------------

static std::string tmpfile(const char* suffix) {
    return std::filesystem::temp_directory_path().string() + "/tgv_t24_" + suffix;
}

// Build a small synthetic cube file from a synthetic depth map.
static std::string make_cube_file(const char* tag,
                                   float r_root, const Vec3f& origin,
                                   float depth_z = 5.0f, int W = 8, int H = 8) {
    DepthMap dm;
    dm.width = W; dm.height = H;
    dm.depths.assign(W * H, depth_z);
    dm.K = Eigen::Matrix3f::Identity();
    dm.K(0,0) = 50; dm.K(1,1) = 50;
    dm.K(0,2) = W/2.0f; dm.K(1,2) = H/2.0f;
    dm.Rt = Eigen::Matrix4f::Identity();

    std::string path = tmpfile(tag);
    build_cubes_from_depthmap(dm, origin, r_root, path);
    return path;
}

// ---- parent_code ------------------------------------------------------------

static void test_parent_code() {
    // ix=5 (odd) at depth 10: fine-grid = 5 << 11 = 10240 (bits 11 and 13 set).
    uint32_t ix_fine = 5u << (21u - 10u);  // 10240
    MortonCode code = morton_encode(ix_fine, 0, 0);
    MortonCode p9   = parent_code(code, 10);  // depth-9 parent: ix=2, fine-grid=2<<12=8192

    uint32_t px, py, pz;
    morton_decode(p9, px, py, pz);
    CHECK(px == (2u << 12u));  // 8192
    CHECK(py == 0u);
    CHECK(pz == 0u);
    CHECK(!(code == p9));  // must differ (odd child's low corner ≠ parent's low corner)

    // Two more levels: depth-9 → depth-8 → depth-7 (ix=0, fine-grid=0)
    MortonCode p7 = parent_code(parent_code(p9, 9), 8);
    uint32_t gx, gy, gz;
    morton_decode(p7, gx, gy, gz);
    CHECK(gx == 0u);  // depth-7 ix = 5>>3 = 0 → fine-grid=0
    CHECK(gy == 0u);
    CHECK(gz == 0u);
}

// ---- face_neighbor ----------------------------------------------------------

static void test_face_neighbor() {
    // Cube at depth 4, cell (1,1,1): fine-grid = (1<<17, 1<<17, 1<<17)
    uint32_t step  = 1u << (21 - 4);  // 131072
    MortonCode c   = morton_encode(step, step, step);

    // +x neighbor: fine-grid x increases by step
    MortonCode nb0 = face_neighbor(c, 0, 4);
    uint32_t nx, ny, nz;
    morton_decode(nb0, nx, ny, nz);
    CHECK(nx == 2 * step);
    CHECK(ny == step);
    CHECK(nz == step);

    // -x neighbor
    MortonCode nb1 = face_neighbor(c, 1, 4);
    morton_decode(nb1, nx, ny, nz);
    CHECK(nx == 0u);

    // Out-of-bounds: cube at (0,0,0), -x direction
    MortonCode corner = morton_encode(0, 0, 0);
    MortonCode oob = face_neighbor(corner, 1, 4);
    CHECK(oob.hi == UINT32_MAX);

    // +x at max cell: cube at depth 4, cell (15,0,0) → fine-grid = 15*step
    MortonCode edge = morton_encode(15 * step, 0, 0);
    MortonCode oob2 = face_neighbor(edge, 0, 4);
    CHECK(oob2.hi == UINT32_MAX);
}

// ---- kway_merge -------------------------------------------------------------

static void test_kway_merge() {
    Vec3f origin(-1.5f, -1.5f, 4.0f);
    float r_root = 3.0f;

    // Two disjoint depth maps (one at z=5, one at z=6)
    std::string a = make_cube_file("merge_a.cubes", r_root, origin, 5.0f);
    std::string b = make_cube_file("merge_b.cubes", r_root, origin, 6.0f);
    std::string merged = tmpfile("merged.cubes");

    kway_merge({a, b}, merged);

    float rr;
    auto cubes = load_cubes(merged, rr);
    CHECK(rr == r_root);
    CHECK(!cubes.empty());

    // Verify sorted
    for (size_t i = 1; i < cubes.size(); ++i)
        CHECK(!(cubes[i].code < cubes[i-1].code));

    float ra, rb;
    auto ca = load_cubes(a, ra);
    auto cb = load_cubes(b, rb);
    CHECK(cubes.size() == ca.size() + cb.size());

    std::filesystem::remove(a);
    std::filesystem::remove(b);
    std::filesystem::remove(merged);
}

// ---- balance_octree ---------------------------------------------------------

static void test_balance_single_cube() {
    // A single cube → no violations → output == input.
    std::string in = tmpfile("bal_in.cubes");
    std::string out = tmpfile("bal_out.cubes");

    // Write a file with one cube manually.
    {
        std::ofstream f(in, std::ios::binary);
        CubeFileHeader hdr;
        hdr.r_root = 10.0f;
        hdr.cube_count = 1;
        f.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        OctreeCube c;
        c.code = morton_encode(0, 0, 0);
        c.depth = 5;
        c.r_c = 0.3f;
        f.write(reinterpret_cast<char*>(&c), sizeof(c));
    }

    balance_octree(in, out);
    float rr;
    auto cubes = load_cubes(out, rr);
    CHECK(cubes.size() >= 1u);

    // Verify 2:1 balance holds: no adjacent pair has depth diff > 1.
    // For a single cube there can't be violations.
    CHECK(cubes.size() == 1u);

    std::filesystem::remove(in);
    std::filesystem::remove(out);
}

static void test_balance_two_adjacent_cubes() {
    // Two cubes at depths 5 and 7 sharing a face → balance must add a bridge at depth 6.
    // Set up so cube A is at depth 5, cell (0,0,0), fine-grid = (0,0,0).
    // Cube B is at depth 7, immediately in the +x direction of A at depth 5.
    //   B must be at fine-grid x = A.x + step_5 = 0 + (1<<16) = 65536.
    //   As depth-7 cell: ix = 65536 / (1<<14) = 4, so cell (4,0,0) at depth 7.

    std::string in  = tmpfile("bal2_in.cubes");
    std::string out = tmpfile("bal2_out.cubes");
    float r_root = 100.0f;

    {
        std::ofstream f(in, std::ios::binary);
        CubeFileHeader hdr;
        hdr.r_root = r_root;
        hdr.cube_count = 2;
        f.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));

        OctreeCube a;
        a.depth = 5;
        a.code  = morton_encode(0, 0, 0);  // cell (0,0,0) at d=5, fine-grid (0,0,0)
        a.r_c   = r_root / (1u << 5);
        f.write(reinterpret_cast<char*>(&a), sizeof(a));

        OctreeCube b;
        b.depth = 7;
        // fine-grid x of B: step at d=5 is 1<<16. cell at d=7: ix = (1<<16)/(1<<14) = 4
        uint32_t step5 = 1u << (21 - 5);  // 1<<16
        b.code  = morton_encode(step5, 0, 0);  // fine-grid x = step5, cell (4,0,0) at d=7
        b.r_c   = r_root / (1u << 7);
        f.write(reinterpret_cast<char*>(&b), sizeof(b));
    }

    balance_octree(in, out);
    float rr;
    auto cubes = load_cubes(out, rr);

    // Must have at least 3 cubes (A, bridge at d=6, B)
    CHECK(cubes.size() >= 3u);

    // Verify sorted
    for (size_t i = 1; i < cubes.size(); ++i)
        CHECK(!(cubes[i].code < cubes[i-1].code));

    // Verify 2:1 balance: check all cubes against all their face neighbors.
    // Build a set for lookup.
    std::set<std::pair<int,MortonCode>> cs;
    for (auto& c : cubes)
        cs.insert({c.depth, c.code});

    bool any_violation = false;
    for (auto& c : cubes) {
        for (int face = 0; face < 6; ++face) {
            MortonCode nb = face_neighbor(c.code, face, c.depth);
            if (nb.hi == UINT32_MAX) continue;

            // Find deepest ancestor of nb that exists in cs
            for (auto& [d, code] : cs) {
                // Is this cube at (d, code) adjacent to c?
                // A cube at depth d2 is adjacent to c (depth d1) via face if
                // it covers the face-neighbor region of c.
                // Simple check: find cubes much coarser than c+1 in nb's region.
                (void)d; (void)code;
            }

            // More direct check: does any cube exist at depth > c.depth+1 adjacent to c?
            // We check if the coarser neighbor at c.depth-1 is missing while something coarser exists.
            if (c.depth >= 2) {
                MortonCode nb_parent = parent_code(nb, c.depth);
                bool same_level = cs.count({c.depth, nb});
                bool coarser    = cs.count({c.depth - 1, nb_parent});
                if (!same_level && !coarser) {
                    // Check for depth-c.depth-2 ancestor
                    for (int d2 = c.depth - 2; d2 >= 0; --d2) {
                        // ancestor of nb at d2
                        uint64_t v = (static_cast<uint64_t>(nb.hi)<<32)|nb.lo;
                        unsigned sh = 3u*(21u-(unsigned)d2);
                        if (sh < 64u) v &= ~((1ULL<<sh)-1ULL);
                        else v = 0;
                        MortonCode anc = {uint32_t(v>>32), uint32_t(v)};
                        if (cs.count({d2, anc})) {
                            any_violation = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    CHECK(!any_violation);

    std::filesystem::remove(in);
    std::filesystem::remove(out);
}

// ---- build_treetop / load_treetop -------------------------------------------

static void test_treetop() {
    Vec3f origin(-1.5f, -1.5f, 4.0f);
    float r_root = 3.0f;

    std::string cubes_path = make_cube_file("treetop.cubes", r_root, origin, 5.0f, 16, 16);
    std::string bal_path   = tmpfile("treetop_bal.cubes");
    std::string tt_path    = tmpfile("treetop.treetop");

    balance_octree(cubes_path, bal_path);
    build_treetop(bal_path, tt_path);

    float rr;
    auto leaves = load_treetop(tt_path, rr);

    CHECK(rr == r_root);
    CHECK(!leaves.empty());

    // Load balanced cubes to verify index coverage.
    float rr2;
    auto bal_cubes = load_cubes(bal_path, rr2);
    uint32_t N = static_cast<uint32_t>(bal_cubes.size());

    // All cube indices must be covered exactly once.
    std::vector<bool> covered(N, false);
    for (auto& leaf : leaves) {
        CHECK(leaf.first_idx <= leaf.last_idx);
        CHECK(leaf.last_idx < N);
        for (uint32_t i = leaf.first_idx; i <= leaf.last_idx; ++i) {
            CHECK(!covered[i]);
            covered[i] = true;
        }
    }
    for (uint32_t i = 0; i < N; ++i)
        CHECK(covered[i]);

    // Each leaf covers ≤ N_CUBES_PER_TREETOP_LEAF cubes (unless it's the only leaf).
    if (leaves.size() > 1) {
        for (auto& leaf : leaves)
            CHECK(leaf.last_idx - leaf.first_idx + 1 <= N_CUBES_PER_TREETOP_LEAF);
    }

    // Leaves are in Morton code order.
    for (size_t i = 1; i < leaves.size(); ++i)
        CHECK(!(leaves[i].code < leaves[i-1].code));

    std::filesystem::remove(cubes_path);
    std::filesystem::remove(bal_path);
    std::filesystem::remove(tt_path);
}

// ---- main -------------------------------------------------------------------

int main() {
    test_parent_code();
    test_face_neighbor();
    test_kway_merge();
    test_balance_single_cube();
    test_balance_two_adjacent_cubes();
    test_treetop();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
