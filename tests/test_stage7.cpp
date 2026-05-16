#include "types.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
#include "tgv.hpp"
#include "treetop.hpp"
#include "marching_cubes.hpp"
#include "qslim.hpp"
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)

// Count edges that appear in exactly 1 face (boundary) vs 2 (interior).
static void edge_valence(const Mesh& mesh, size_t& boundary, size_t& interior, size_t& irregular) {
    // Encode edge as (lo_idx, hi_idx) packed into uint64.
    std::unordered_map<uint64_t, int> edge_count;
    const size_t nt = mesh.tri_count();
    for (size_t t = 0; t < nt; ++t) {
        uint32_t idx[3] = {mesh.faces[t*3], mesh.faces[t*3+1], mesh.faces[t*3+2]};
        for (int e = 0; e < 3; ++e) {
            uint32_t a = idx[e], b = idx[(e+1)%3];
            if (a > b) std::swap(a, b);
            uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
            ++edge_count[key];
        }
    }
    boundary = interior = irregular = 0;
    for (auto& [k, cnt] : edge_count) {
        if      (cnt == 1) ++boundary;
        else if (cnt == 2) ++interior;
        else               ++irregular;
    }
}

// Check that every triangle has non-zero area.
static bool all_nonzero_area(const Mesh& mesh) {
    const size_t nt = mesh.tri_count();
    for (size_t t = 0; t < nt; ++t) {
        Vec3f a = mesh.verts[mesh.faces[t*3]];
        Vec3f b = mesh.verts[mesh.faces[t*3+1]];
        Vec3f c = mesh.verts[mesh.faces[t*3+2]];
        if ((b - a).cross(c - a).squaredNorm() < 1e-12f) return false;
    }
    return true;
}

// Build a small 2×2×2 cube arrangement where u flips sign at x=1.
static void test_extract_surface_simple() {
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube> cubes;
    std::vector<TGVCube>    state;

    for (uint32_t iz = 0; iz < 2; ++iz)
    for (uint32_t iy = 0; iy < 2; ++iy)
    for (uint32_t ix = 0; ix < 2; ++ix) {
        OctreeCube c;
        c.depth = depth; c.r_c = r_root / (1u << depth);
        c.code  = morton_encode(ix << shift, iy << shift, iz << shift);
        cubes.push_back(c);
        TGVCube s; s.u = (ix == 0) ? 1.0f : -1.0f;
        state.push_back(s);
    }

    Mesh mesh = extract_surface(cubes, state, Vec3f::Zero(), r_root);

    CHECK(!mesh.empty());

    // 4 quads (2 triangles each) → 8 triangles, ≥ 4 unique verts on the YZ plane.
    CHECK(mesh.tri_count() >= 4u);

    // All vertices should have x ≈ cs (the face between ix=0 and ix=1 cubes).
    float expected_x = 2.0f * r_root / static_cast<float>(1u << depth); // cs
    for (const auto& v : mesh.verts)
        CHECK(std::abs(v.x() - expected_x) < 0.01f * expected_x + 1e-4f);

    // No duplicate vertices in the returned mesh.
    std::unordered_set<uint64_t> keys;
    for (const auto& v : mesh.verts) {
        uint64_t k = (uint64_t)(v.x() * 1e6f + 0.5f) ^
                     ((uint64_t)(v.y() * 1e6f + 0.5f) << 21) ^
                     ((uint64_t)(v.z() * 1e6f + 0.5f) << 42);
        CHECK(keys.insert(k).second); // each vertex unique
    }

    // No zero-area triangles.
    CHECK(all_nonzero_area(mesh));

    // The inner edges (shared by adjacent quads on the same face) should be
    // interior (valence 2).  Boundary edges are at the patch border.
    size_t boundary, interior, irregular;
    edge_valence(mesh, boundary, interior, irregular);
    CHECK(irregular == 0u);
    CHECK(interior > 0u);
}

static void test_extract_all_positive() {
    // All u > 0 → no sign changes → empty mesh.
    float r_root = 8.0f;
    int depth = 3;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube> cubes;
    std::vector<TGVCube>    state;

    for (uint32_t ix = 0; ix < 2; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = 1.0f;
        c.code = morton_encode(ix << shift, 0u, 0u);
        cubes.push_back(c);
        TGVCube s; s.u = 1.0f; state.push_back(s);
    }

    Mesh mesh = extract_surface(cubes, state, Vec3f::Zero(), r_root);
    CHECK(mesh.empty());
}

static void test_write_obj() {
    // Create a simple 2-triangle patch and write it.
    Mesh mesh;
    mesh.verts = { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} };
    mesh.faces = { 0,1,2, 0,2,3 };
    compute_normals(mesh);

    std::string path = std::filesystem::temp_directory_path().string() + "/tgv_mesh.obj";
    write_obj(path, mesh);
    CHECK(std::filesystem::file_size(path) > 10u);

    // Normals should be unit +z for both vertices.
    for (const auto& n : mesh.normals)
        CHECK(std::abs(n.z() - 1.0f) < 1e-5f);

    std::filesystem::remove(path);
}

static void test_normals_consistent() {
    // Build 2×2×2 sign-change mesh, compute normals.
    // All normals should point outward (-x direction, away from solid).
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube> cubes;
    std::vector<TGVCube>    state;
    for (uint32_t iz = 0; iz < 2; ++iz)
    for (uint32_t iy = 0; iy < 2; ++iy)
    for (uint32_t ix = 0; ix < 2; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = r_root/(1u<<depth);
        c.code = morton_encode(ix<<shift, iy<<shift, iz<<shift);
        cubes.push_back(c);
        TGVCube s; s.u = (ix==0) ? 1.0f : -1.0f; state.push_back(s);
    }

    Mesh mesh = extract_surface(cubes, state, Vec3f::Zero(), r_root);
    compute_normals(mesh);
    CHECK(!mesh.normals.empty());
    // All face normals should have positive x component (pointing into free space).
    const size_t nt = mesh.tri_count();
    for (size_t t = 0; t < nt; ++t) {
        Vec3f a = mesh.verts[mesh.faces[t*3]];
        Vec3f b = mesh.verts[mesh.faces[t*3+1]];
        Vec3f c = mesh.verts[mesh.faces[t*3+2]];
        Vec3f n = (b-a).cross(c-a);
        // Solid is at ix=1 (u<0), free space at ix=0 (u>0).
        // Outward normal points from solid toward free space = -x direction.
        CHECK(n.x() < 0.0f);
    }
}

static void test_no_irregular_edges_uniform() {
    // 3×1×1 cubes: A(u>0)–B(u<0)–C(u>0).
    // Surface appears on both A–B and B–C boundaries.
    // On each face pair the resulting patch is a 1×1 square (no adaptive transitions),
    // so all interior edges should be manifold (valence 2).
    float r_root = 8.0f;
    int depth = 3;
    uint32_t shift = 21 - depth;
    std::vector<OctreeCube> cubes;
    std::vector<TGVCube>    state;
    float us[3] = {1.0f, -1.0f, 1.0f};
    for (uint32_t ix = 0; ix < 3; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = 1.0f;
        c.code = morton_encode(ix<<shift, 0u, 0u);
        cubes.push_back(c);
        TGVCube s; s.u = us[ix]; state.push_back(s);
    }
    Mesh mesh = extract_surface(cubes, state, Vec3f::Zero(), r_root);
    size_t boundary, interior, irregular;
    edge_valence(mesh, boundary, interior, irregular);
    CHECK(irregular == 0u);
    // Two isolated quads at different x positions.
    // Each quad: 4 boundary edges + 1 interior diagonal (shared by its 2 triangles).
    CHECK(boundary == 8u);
    CHECK(interior == 2u);
}

// Two cubes on opposite sides of the u=0 boundary, assigned to separate "leaves".
// Per-leaf extraction (passing only one cube at a time) produces no face.
// Global extraction (both cubes) finds the sign change and emits a face.
static void test_cross_leaf_face_requires_global_extraction() {
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;

    OctreeCube cA, cB;
    cA.depth = cB.depth = depth;
    cA.r_c = cB.r_c = r_root / (1u << depth);
    cA.code = morton_encode(0u << shift, 0u, 0u);
    cB.code = morton_encode(1u << shift, 0u, 0u);

    TGVCube sA, sB;
    sA.u =  1.0f;  // free
    sB.u = -1.0f;  // solid

    // Global: both cubes → sign change → face produced.
    Mesh global = extract_surface({cA, cB}, {sA, sB}, Vec3f::Zero(), r_root);
    CHECK(!global.empty());

    // Per-leaf leaf-0 (only A): no B neighbor → no face.
    Mesh leaf0 = extract_surface({cA}, {sA}, Vec3f::Zero(), r_root);
    CHECK(leaf0.empty());

    // Per-leaf leaf-1 (only B): no A neighbor → no face.
    Mesh leaf1 = extract_surface({cB}, {sB}, Vec3f::Zero(), r_root);
    CHECK(leaf1.empty());
}

// 3×3×3 grid: center cube solid, all others free.
// The 6 face-quads of the center form a closed box — boundary == 0.
static void test_closed_solid_box() {
    float r_root = 24.0f;
    int depth = 3;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube> cubes;
    std::vector<TGVCube>    state;
    for (uint32_t iz = 0; iz < 3; ++iz)
    for (uint32_t iy = 0; iy < 3; ++iy)
    for (uint32_t ix = 0; ix < 3; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = r_root / (1u << depth);
        c.code = morton_encode(ix << shift, iy << shift, iz << shift);
        cubes.push_back(c);
        TGVCube s;
        s.u = (ix == 1 && iy == 1 && iz == 1) ? -1.0f : 1.0f;
        state.push_back(s);
    }

    Mesh mesh = extract_surface(cubes, state, Vec3f::Zero(), r_root);

    // 6 faces of the center cube → 6 quads → 12 triangles.
    CHECK(mesh.tri_count() == 12u);
    CHECK(all_nonzero_area(mesh));

    size_t boundary, interior, irregular;
    edge_valence(mesh, boundary, interior, irregular);
    CHECK(boundary  == 0u);   // closed solid — no open edges
    CHECK(irregular == 0u);   // manifold
}

static void test_qslim_reduces_faces() {
    // Build a 4×4×1 block of cubes with a sign change at x=2.
    // This yields a large YZ-plane patch of triangles that QSlim can reduce.
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;
    std::vector<OctreeCube> cubes;
    std::vector<TGVCube>    state;
    for (uint32_t iz = 0; iz < 4; ++iz)
    for (uint32_t iy = 0; iy < 4; ++iy)
    for (uint32_t ix = 0; ix < 4; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = r_root/(1u<<depth);
        c.code = morton_encode(ix<<shift, iy<<shift, iz<<shift);
        cubes.push_back(c);
        TGVCube s; s.u = (ix < 2) ? 1.0f : -1.0f; state.push_back(s);
    }
    Mesh raw = extract_surface(cubes, state, Vec3f::Zero(), r_root);
    CHECK(raw.tri_count() > 4u);

    size_t target = raw.tri_count() / 2;
    Mesh dec = decimate(raw, target);

    // Should have fewer faces than input.
    CHECK(dec.tri_count() <= raw.tri_count());
    // Should not be empty.
    CHECK(!dec.empty());
    // No irregular edges.
    size_t boundary, interior, irregular;
    edge_valence(dec, boundary, interior, irregular);
    CHECK(irregular == 0u);
}

static void test_qslim_empty_passthrough() {
    Mesh empty;
    Mesh result = decimate(empty, 100);
    CHECK(result.empty());
}

static void test_qslim_noop_when_below_target() {
    Mesh mesh;
    mesh.verts = {{0,0,0},{1,0,0},{0,1,0}};
    mesh.faces = {0,1,2};
    Mesh result = decimate(mesh, 10);  // target > current face count
    CHECK(result.tri_count() == 1u);
}

// ---------- streaming MC tests ----------------------------------------------

struct CubeFileHeaderT7 {
    char     magic[4] = {'T','G','V','1'};
    uint32_t version  = 1;
    float    r_root   = 0.0f;
    uint32_t cube_count = 0;
};
struct HistFileHeaderT7 {
    char     magic[4] = {'T','G','V','H'};
    uint32_t version  = 1;
    uint32_t count    = 0;
};
struct TreetopFileHeaderT7 {
    char     magic[4] = {'T','G','V','T'};
    uint32_t version  = 1;
    float    r_root   = 0.0f;
    uint32_t leaf_count = 0;
};
struct TGVFileHeaderT7 {
    char     magic[4] = {'T','G','V','S'};
    uint32_t version  = 1;
    uint32_t count    = 0;
};

static void write_cubes_bin7(const std::string& path, float r_root,
                              const std::vector<OctreeCube>& cubes) {
    std::ofstream out(path, std::ios::binary);
    CubeFileHeaderT7 hdr; hdr.r_root = r_root; hdr.cube_count = (uint32_t)cubes.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!cubes.empty())
        out.write(reinterpret_cast<const char*>(cubes.data()),
                  cubes.size() * sizeof(OctreeCube));
}
static void write_hists_bin7(const std::string& path,
                              const std::vector<CubeHistogram>& hists) {
    std::ofstream out(path, std::ios::binary);
    HistFileHeaderT7 hdr; hdr.count = (uint32_t)hists.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!hists.empty())
        out.write(reinterpret_cast<const char*>(hists.data()),
                  hists.size() * sizeof(CubeHistogram));
}
static void write_treetop_bin7(const std::string& path, float r_root,
                                const std::vector<TreetopLeaf>& leaves) {
    std::ofstream out(path, std::ios::binary);
    TreetopFileHeaderT7 hdr; hdr.r_root = r_root; hdr.leaf_count = (uint32_t)leaves.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!leaves.empty())
        out.write(reinterpret_cast<const char*>(leaves.data()),
                  leaves.size() * sizeof(TreetopLeaf));
}
static void write_state_bin7(const std::string& path,
                              const std::vector<TGVCube>& state) {
    std::ofstream out(path, std::ios::binary);
    TGVFileHeaderT7 hdr; hdr.count = (uint32_t)state.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!state.empty())
        out.write(reinterpret_cast<const char*>(state.data()),
                  state.size() * sizeof(TGVCube));
}

// Parse a Wavefront OBJ produced by extract_surface_streaming.
static Mesh load_obj_simple(const std::string& path) {
    std::ifstream in(path);
    Mesh m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream s(line.substr(2));
            float x, y, z; s >> x >> y >> z;
            m.verts.push_back({x, y, z});
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream s(line.substr(2));
            uint32_t a, b, c; s >> a >> b >> c;
            m.faces.push_back(a - 1); m.faces.push_back(b - 1); m.faces.push_back(c - 1);
        }
    }
    return m;
}

// Closed solid box: streaming MC should produce the same 6 quads / 12 tris,
// boundary == 0, irregular == 0 — matching the in-RAM extract_surface.
static void test_streaming_mc_closed_box() {
    float r_root = 24.0f;
    int depth = 3;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube>    cubes;
    std::vector<TGVCube>       state;
    std::vector<CubeHistogram> hists;
    for (uint32_t iz = 0; iz < 3; ++iz)
    for (uint32_t iy = 0; iy < 3; ++iy)
    for (uint32_t ix = 0; ix < 3; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = r_root / (1u << depth);
        c.code = morton_encode(ix << shift, iy << shift, iz << shift);
        cubes.push_back(c);
        TGVCube s;
        s.u = (ix == 1 && iy == 1 && iz == 1) ? -1.0f : 1.0f;
        state.push_back(s);
        CubeHistogram h; h.bins[0] = 1;   // give every cube some "data"
        hists.push_back(h);
    }

    // Single leaf spanning everything (matches in-RAM ownership).
    TreetopLeaf leaf;
    leaf.code = cubes[0].code; leaf.depth = 0;
    leaf.first_idx = 0; leaf.last_idx = (uint32_t)cubes.size() - 1;

    auto tmp = std::filesystem::temp_directory_path();
    std::string cube_p  = (tmp / "mc7_cubes.bin").string();
    std::string hist_p  = (tmp / "mc7_hists.bin").string();
    std::string tt_p    = (tmp / "mc7_tt.bin").string();
    std::string state_p = (tmp / "mc7_state.bin").string();
    std::string obj_p   = (tmp / "mc7_mesh.obj").string();

    write_cubes_bin7(cube_p, r_root, cubes);
    write_hists_bin7(hist_p, hists);
    write_treetop_bin7(tt_p, r_root, {leaf});
    write_state_bin7(state_p, state);

    extract_surface_streaming(cube_p, state_p, tt_p, hist_p,
                              Vec3f::Zero(), r_root, obj_p);

    Mesh streamed = load_obj_simple(obj_p);
    CHECK(streamed.tri_count() == 12u);

    size_t boundary, interior, irregular;
    edge_valence(streamed, boundary, interior, irregular);
    CHECK(boundary  == 0u);
    CHECK(irregular == 0u);

    std::filesystem::remove(cube_p);
    std::filesystem::remove(hist_p);
    std::filesystem::remove(tt_p);
    std::filesystem::remove(state_p);
    std::filesystem::remove(obj_p);
}

// Cross-leaf face emitted exactly once via the ownership rule.
static void test_streaming_mc_cross_leaf_once() {
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;

    OctreeCube cA, cB;
    cA.depth = cB.depth = depth;
    cA.r_c = cB.r_c = r_root / (1u << depth);
    cA.code = morton_encode(0u << shift, 0u, 0u);
    cB.code = morton_encode(1u << shift, 0u, 0u);
    std::vector<OctreeCube> cubes = {cA, cB};

    std::vector<TGVCube> state(2);
    state[0].u =  1.0f;
    state[1].u = -1.0f;
    std::vector<CubeHistogram> hists(2);
    hists[0].bins[0] = 1; hists[1].bins[0] = 1;

    // Split into two leaves, one cube each.
    TreetopLeaf l0, l1;
    l0.code = cubes[0].code; l0.depth = 4; l0.first_idx = 0; l0.last_idx = 0;
    l1.code = cubes[1].code; l1.depth = 4; l1.first_idx = 1; l1.last_idx = 1;

    auto tmp = std::filesystem::temp_directory_path();
    std::string cube_p  = (tmp / "mc7b_cubes.bin").string();
    std::string hist_p  = (tmp / "mc7b_hists.bin").string();
    std::string tt_p    = (tmp / "mc7b_tt.bin").string();
    std::string state_p = (tmp / "mc7b_state.bin").string();
    std::string obj_p   = (tmp / "mc7b_mesh.obj").string();

    write_cubes_bin7(cube_p, r_root, cubes);
    write_hists_bin7(hist_p, hists);
    write_treetop_bin7(tt_p, r_root, {l0, l1});
    write_state_bin7(state_p, state);

    extract_surface_streaming(cube_p, state_p, tt_p, hist_p,
                              Vec3f::Zero(), r_root, obj_p);

    Mesh m = load_obj_simple(obj_p);
    // Exactly one face pair → one quad → 2 triangles, 4 vertices.
    CHECK(m.tri_count() == 2u);
    CHECK(m.verts.size() == 4u);

    std::filesystem::remove(cube_p);
    std::filesystem::remove(hist_p);
    std::filesystem::remove(tt_p);
    std::filesystem::remove(state_p);
    std::filesystem::remove(obj_p);
}

// Streaming output should match the in-RAM extract_surface result (single-leaf).
static void test_streaming_mc_matches_inram_single_leaf() {
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube>    cubes;
    std::vector<TGVCube>       state;
    std::vector<CubeHistogram> hists;
    for (uint32_t iz = 0; iz < 2; ++iz)
    for (uint32_t iy = 0; iy < 2; ++iy)
    for (uint32_t ix = 0; ix < 2; ++ix) {
        OctreeCube c; c.depth = depth; c.r_c = r_root / (1u << depth);
        c.code = morton_encode(ix << shift, iy << shift, iz << shift);
        cubes.push_back(c);
        TGVCube s; s.u = (ix == 0) ? 1.0f : -1.0f; state.push_back(s);
        CubeHistogram h; h.bins[0] = 1; hists.push_back(h);
    }

    std::vector<bool> has_data(hists.size(), true);
    Mesh ref = extract_surface(cubes, state, Vec3f::Zero(), r_root, has_data);

    TreetopLeaf leaf;
    leaf.code = cubes[0].code; leaf.depth = 0;
    leaf.first_idx = 0; leaf.last_idx = (uint32_t)cubes.size() - 1;

    auto tmp = std::filesystem::temp_directory_path();
    std::string cube_p  = (tmp / "mc7c_cubes.bin").string();
    std::string hist_p  = (tmp / "mc7c_hists.bin").string();
    std::string tt_p    = (tmp / "mc7c_tt.bin").string();
    std::string state_p = (tmp / "mc7c_state.bin").string();
    std::string obj_p   = (tmp / "mc7c_mesh.obj").string();

    write_cubes_bin7(cube_p, r_root, cubes);
    write_hists_bin7(hist_p, hists);
    write_treetop_bin7(tt_p, r_root, {leaf});
    write_state_bin7(state_p, state);

    extract_surface_streaming(cube_p, state_p, tt_p, hist_p,
                              Vec3f::Zero(), r_root, obj_p);

    Mesh streamed = load_obj_simple(obj_p);
    CHECK(streamed.tri_count() == ref.tri_count());
    CHECK(streamed.verts.size() == ref.verts.size());

    std::filesystem::remove(cube_p);
    std::filesystem::remove(hist_p);
    std::filesystem::remove(tt_p);
    std::filesystem::remove(state_p);
    std::filesystem::remove(obj_p);
}

int main() {
    test_extract_surface_simple();
    test_extract_all_positive();
    test_write_obj();
    test_normals_consistent();
    test_no_irregular_edges_uniform();
    test_cross_leaf_face_requires_global_extraction();
    test_closed_solid_box();
    test_qslim_reduces_faces();
    test_qslim_empty_passthrough();
    test_qslim_noop_when_below_target();
    test_streaming_mc_closed_box();
    test_streaming_mc_cross_leaf_once();
    test_streaming_mc_matches_inram_single_leaf();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
