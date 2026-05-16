#include "types.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
#include "tgv.hpp"
#include "treetop.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)
#define CHECK_NEAR(a, b, eps) CHECK(std::abs((float)(a)-(float)(b)) <= (float)(eps))

// Build a small 3-cube chain at depth 5: A–B–C along x.
// A has strong "in-front" signal (u→+1), C has strong "behind" signal (u→-1).
// After TGV, B (with weak or zero data) should have u between A and C.
static void test_tgv_smoothing() {
    float r_root = 32.0f;
    uint32_t shift = 21 - 5;

    auto make_cube = [&](uint32_t ix, uint32_t iy, uint32_t iz) {
        OctreeCube c;
        c.depth = 5;
        c.r_c   = r_root / (1u << 5);  // 1.0
        c.code  = morton_encode(ix << shift, iy << shift, iz << shift);
        return c;
    };

    OctreeCube cA = make_cube(0, 0, 0);
    OctreeCube cB = make_cube(1, 0, 0);
    OctreeCube cC = make_cube(2, 0, 0);

    std::vector<OctreeCube> cubes = {cA, cB, cC};

    // Histograms: A mostly in "in-front" bins (4-7), C mostly in "behind" bins (0-3).
    CubeHistogram hA, hB, hC;
    hA.bins[6] = 50;  // strong positive
    hC.bins[1] = 50;  // strong negative
    // hB is empty (no observations)

    std::vector<CubeHistogram> hists = {hA, hB, hC};

    TGVParams p;
    p.iters  = 50;
    p.alpha0 = 0.1f;
    p.alpha1 = 0.2f;

    auto state = tgv_minimize(cubes, hists, Vec3f::Zero(), r_root, p);

    CHECK(state.size() == 3u);

    // A should remain positive, C negative.
    CHECK(state[0].u > 0.0f);
    CHECK(state[2].u < 0.0f);
    // B (between A and C, no data) should be between them.
    CHECK(state[1].u > state[2].u);
    CHECK(state[1].u < state[0].u);
}

static void test_tgv_constant() {
    // All cubes with identical histogram → u should stay constant after TGV.
    float r_root = 16.0f;
    uint32_t shift = 21 - 4;

    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;

    for (uint32_t ix = 0; ix < 4; ++ix) {
        OctreeCube c;
        c.depth = 4;
        c.r_c   = r_root / (1u << 4);
        c.code  = morton_encode(ix << shift, 0u, 0u);
        cubes.push_back(c);

        CubeHistogram h;
        h.bins[6] = 10;  // all positive, same
        hists.push_back(h);
    }

    TGVParams p; p.iters = 20;
    auto state = tgv_minimize(cubes, hists, Vec3f::Zero(), r_root, p);

    // All u should be similar (close to median of bin 6 = -1 + 6.5/4 = 0.625).
    for (auto& s : state)
        CHECK_NEAR(s.u, 0.625f, 0.15f);
}

static void test_tgv_save_load() {
    std::vector<TGVCube> state(3);
    state[0].u = 1.0f; state[0].v[0] = 0.1f;
    state[1].u = 0.0f;
    state[2].u = -1.0f; state[2].p[2] = 0.5f;

    std::string path = std::filesystem::temp_directory_path().string() + "/tgv_test_state.s";
    save_tgv(path, state);
    auto s2 = load_tgv(path);
    CHECK(s2.size() == 3u);
    CHECK_NEAR(s2[0].u, 1.0f, 1e-6f);
    CHECK_NEAR(s2[2].p[2], 0.5f, 1e-6f);
    std::filesystem::remove(path);
}

// Build a 3-cube chain (same as test_tgv_smoothing) as a shared helper.
static void make_3cube_chain(std::vector<OctreeCube>& cubes,
                              std::vector<CubeHistogram>& hists,
                              float r_root)
{
    const uint32_t shift = 21 - 5;
    auto make_cube = [&](uint32_t ix) {
        OctreeCube c;
        c.depth = 5;
        c.r_c   = r_root / (1u << 5);
        c.code  = morton_encode(ix << shift, 0u, 0u);
        return c;
    };
    cubes = { make_cube(0), make_cube(1), make_cube(2) };

    CubeHistogram hA, hB, hC;
    hA.bins[6] = 50;   // strong positive
    hC.bins[1] = 50;   // strong negative
    hists = { hA, hB, hC };
}

// With a single leaf covering all cubes, oc should equal regular tgv_minimize.
static void test_tgv_oc_matches_regular() {
    float r_root = 32.0f;
    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;
    make_3cube_chain(cubes, hists, r_root);

    TreetopLeaf leaf;
    leaf.code       = cubes[0].code;
    leaf.depth      = 5;
    leaf.first_idx  = 0;
    leaf.last_idx   = 2;

    TGVParams p; p.iters = 50; p.alpha0 = 0.1f; p.alpha1 = 0.2f;

    auto s1 = tgv_minimize   (cubes, hists, Vec3f::Zero(), r_root, p);
    auto s2 = tgv_minimize_oc(cubes, hists, {leaf}, Vec3f::Zero(), r_root, p);

    CHECK(s2.size() == 3u);
    for (size_t i = 0; i < 3; ++i)
        CHECK_NEAR(s1[i].u, s2[i].u, 1e-5f);
}

// With two leaves, each leaf should still converge to the correct polarity.
// Leaf 0 = {A, B}, Leaf 1 = {C}. A: positive, B: empty, C: negative.
static void test_tgv_oc_frozen_border_polarity() {
    float r_root = 32.0f;
    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;
    make_3cube_chain(cubes, hists, r_root);

    TreetopLeaf l0, l1;
    l0.code = cubes[0].code; l0.depth = 5; l0.first_idx = 0; l0.last_idx = 1;
    l1.code = cubes[2].code; l1.depth = 5; l1.first_idx = 2; l1.last_idx = 2;

    TGVParams p; p.iters = 50; p.alpha0 = 0.1f; p.alpha1 = 0.2f;
    auto s = tgv_minimize_oc(cubes, hists, {l0, l1}, Vec3f::Zero(), r_root, p);

    CHECK(s.size() == 3u);
    CHECK(s[0].u > 0.0f);          // A: positive data
    CHECK(s[2].u < 0.0f);          // C: negative data
    CHECK(s[1].u > s[2].u);        // B between A and C
    CHECK(s[1].u < s[0].u);
}

// Empty cube list: oc should return empty state without crashing.
static void test_tgv_oc_empty() {
    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;
    std::vector<TreetopLeaf> leaves;
    auto s = tgv_minimize_oc(cubes, hists, leaves, Vec3f::Zero(), 1.0f);
    CHECK(s.empty());
}

// ---------- Streaming TGV tests ----------------------------------------------

// Match the on-disk headers used by Stage 2/3/4/5.
struct CubeFileHeaderT {
    char     magic[4] = {'T','G','V','1'};
    uint32_t version  = 1;
    float    r_root   = 0.0f;
    uint32_t cube_count = 0;
};
struct HistFileHeaderT {
    char     magic[4] = {'T','G','V','H'};
    uint32_t version  = 1;
    uint32_t count    = 0;
};
struct TreetopFileHeaderT {
    char     magic[4] = {'T','G','V','T'};
    uint32_t version  = 1;
    float    r_root   = 0.0f;
    uint32_t leaf_count = 0;
};

static void write_cubes_bin(const std::string& path, float r_root,
                             const std::vector<OctreeCube>& cubes) {
    std::ofstream out(path, std::ios::binary);
    CubeFileHeaderT hdr;
    hdr.r_root = r_root;
    hdr.cube_count = (uint32_t)cubes.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!cubes.empty())
        out.write(reinterpret_cast<const char*>(cubes.data()),
                  cubes.size() * sizeof(OctreeCube));
}
static void write_hists_bin(const std::string& path,
                             const std::vector<CubeHistogram>& hists) {
    std::ofstream out(path, std::ios::binary);
    HistFileHeaderT hdr;
    hdr.count = (uint32_t)hists.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!hists.empty())
        out.write(reinterpret_cast<const char*>(hists.data()),
                  hists.size() * sizeof(CubeHistogram));
}
static void write_treetop_bin(const std::string& path, float r_root,
                               const std::vector<TreetopLeaf>& leaves) {
    std::ofstream out(path, std::ios::binary);
    TreetopFileHeaderT hdr;
    hdr.r_root = r_root;
    hdr.leaf_count = (uint32_t)leaves.size();
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!leaves.empty())
        out.write(reinterpret_cast<const char*>(leaves.data()),
                  leaves.size() * sizeof(TreetopLeaf));
}

// Streaming with a single leaf covering all cubes should match in-memory
// tgv_minimize on the 3-cube chain (no frozen-border activation).
static void test_tgv_streaming_matches_inmemory() {
    float r_root = 32.0f;
    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;
    make_3cube_chain(cubes, hists, r_root);

    TreetopLeaf leaf;
    leaf.code = cubes[0].code; leaf.depth = 5;
    leaf.first_idx = 0; leaf.last_idx = 2;

    auto tmp = std::filesystem::temp_directory_path();
    std::string cube_p   = (tmp / "tgv_stream_cubes.bin").string();
    std::string hist_p   = (tmp / "tgv_stream_hists.bin").string();
    std::string tt_p     = (tmp / "tgv_stream_tt.bin").string();
    std::string state_p  = (tmp / "tgv_stream_state.bin").string();
    write_cubes_bin(cube_p, r_root, cubes);
    write_hists_bin(hist_p, hists);
    write_treetop_bin(tt_p, r_root, {leaf});

    TGVParams p; p.iters = 50; p.alpha0 = 0.1f; p.alpha1 = 0.2f;
    auto s_ref = tgv_minimize(cubes, hists, Vec3f::Zero(), r_root, p);
    tgv_minimize_streaming(cube_p, hist_p, tt_p, Vec3f::Zero(), r_root, state_p, p);
    auto s_str = load_tgv(state_p);

    CHECK(s_str.size() == 3u);
    for (size_t i = 0; i < 3; ++i)
        CHECK_NEAR(s_ref[i].u, s_str[i].u, 1e-4f);

    std::filesystem::remove(cube_p);
    std::filesystem::remove(hist_p);
    std::filesystem::remove(tt_p);
    std::filesystem::remove(state_p);
}

// Two-leaf streaming: polarity preserved across leaf boundary.
static void test_tgv_streaming_two_leaves() {
    float r_root = 32.0f;
    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;
    make_3cube_chain(cubes, hists, r_root);

    TreetopLeaf l0, l1;
    l0.code = cubes[0].code; l0.depth = 5; l0.first_idx = 0; l0.last_idx = 1;
    l1.code = cubes[2].code; l1.depth = 5; l1.first_idx = 2; l1.last_idx = 2;

    auto tmp = std::filesystem::temp_directory_path();
    std::string cube_p   = (tmp / "tgv_stream2_cubes.bin").string();
    std::string hist_p   = (tmp / "tgv_stream2_hists.bin").string();
    std::string tt_p     = (tmp / "tgv_stream2_tt.bin").string();
    std::string state_p  = (tmp / "tgv_stream2_state.bin").string();
    write_cubes_bin(cube_p, r_root, cubes);
    write_hists_bin(hist_p, hists);
    write_treetop_bin(tt_p, r_root, {l0, l1});

    TGVParams p; p.iters = 50; p.alpha0 = 0.1f; p.alpha1 = 0.2f;
    tgv_minimize_streaming(cube_p, hist_p, tt_p, Vec3f::Zero(), r_root, state_p, p);
    auto s = load_tgv(state_p);

    CHECK(s.size() == 3u);
    CHECK(s[0].u > 0.0f);
    CHECK(s[2].u < 0.0f);
    CHECK(s[1].u > s[2].u);
    CHECK(s[1].u < s[0].u);

    std::filesystem::remove(cube_p);
    std::filesystem::remove(hist_p);
    std::filesystem::remove(tt_p);
    std::filesystem::remove(state_p);
}

// State file written by streaming is readable via the regular load_tgv path.
static void test_tgv_streaming_save_load() {
    float r_root = 32.0f;
    std::vector<OctreeCube> cubes;
    std::vector<CubeHistogram> hists;
    make_3cube_chain(cubes, hists, r_root);
    TreetopLeaf leaf;
    leaf.code = cubes[0].code; leaf.depth = 5;
    leaf.first_idx = 0; leaf.last_idx = 2;

    auto tmp = std::filesystem::temp_directory_path();
    std::string cube_p   = (tmp / "tgv_stream3_cubes.bin").string();
    std::string hist_p   = (tmp / "tgv_stream3_hists.bin").string();
    std::string tt_p     = (tmp / "tgv_stream3_tt.bin").string();
    std::string state_p  = (tmp / "tgv_stream3_state.bin").string();
    write_cubes_bin(cube_p, r_root, cubes);
    write_hists_bin(hist_p, hists);
    write_treetop_bin(tt_p, r_root, {leaf});

    TGVParams p; p.iters = 20;
    tgv_minimize_streaming(cube_p, hist_p, tt_p, Vec3f::Zero(), r_root, state_p, p);

    auto s = load_tgv(state_p);
    CHECK(s.size() == 3u);
    CHECK(std::isfinite(s[0].u));
    CHECK(std::isfinite(s[1].u));
    CHECK(std::isfinite(s[2].u));

    std::filesystem::remove(cube_p);
    std::filesystem::remove(hist_p);
    std::filesystem::remove(tt_p);
    std::filesystem::remove(state_p);
}

int main() {
    test_tgv_smoothing();
    test_tgv_constant();
    test_tgv_save_load();
    test_tgv_oc_matches_regular();
    test_tgv_oc_frozen_border_polarity();
    test_tgv_oc_empty();
    test_tgv_streaming_matches_inmemory();
    test_tgv_streaming_two_leaves();
    test_tgv_streaming_save_load();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
