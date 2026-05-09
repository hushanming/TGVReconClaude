#include "types.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
#include "tgv.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
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

int main() {
    test_tgv_smoothing();
    test_tgv_constant();
    test_tgv_save_load();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
