#include "types.hpp"
#include "depth_map.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
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

// ---- DepthMipmap construction -----------------------------------------------

static void test_mipmap_build() {
    DepthMap dm;
    dm.width = 8; dm.height = 8;
    dm.depths.assign(64, 10.0f);
    dm.K = Eigen::Matrix3f::Identity();
    dm.K(0,0) = 100; dm.K(1,1) = 100; dm.K(0,2) = 4; dm.K(1,2) = 4;
    dm.Rt = Eigen::Matrix4f::Identity();

    DepthMipmap mip;
    mip.build(dm, false);

    CHECK(mip.levels.size() >= 4u);  // 8x8 → 4x4 → 2x2 → 1x1
    CHECK(mip.levels[0].width  == 8);
    CHECK(mip.levels[1].width  == 4);
    CHECK(mip.levels[2].width  == 2);

    // All depths should remain 10.0 after averaging constants.
    CHECK_NEAR(mip.depth_at(0, 3, 3), 10.0f, 1e-4f);
    CHECK_NEAR(mip.depth_at(1, 2, 2), 10.0f, 1e-4f);
    CHECK_NEAR(mip.depth_at(2, 1, 1), 10.0f, 1e-4f);
}

// ---- DepthMipmap::project ---------------------------------------------------

static void test_mipmap_project() {
    DepthMap dm;
    dm.width = 32; dm.height = 32;
    dm.depths.assign(32*32, 5.0f);
    dm.K = Eigen::Matrix3f::Identity();
    dm.K(0,0) = 100; dm.K(1,1) = 100; dm.K(0,2) = 16; dm.K(1,2) = 16;
    dm.Rt = Eigen::Matrix4f::Identity();  // camera at origin, z-forward

    DepthMipmap mip;
    mip.build(dm, false);

    // World point directly in front of camera.
    Vec3f P{0.0f, 0.0f, 5.0f};
    int level, u, v;
    float cam_z;
    bool ok = mip.project(P, 0.04f, level, u, v, cam_z);  // footprint=100*0.04/5=0.8 → level 0
    CHECK(ok);
    CHECK_NEAR(cam_z, 5.0f, 1e-4f);
    CHECK(level == 0);  // tiny r_c → level 0

    // Point behind camera → should fail.
    Vec3f behind{0.0f, 0.0f, -1.0f};
    ok = mip.project(behind, 0.1f, level, u, v, cam_z);
    CHECK(!ok);

    // Point off-screen → should fail.
    Vec3f offscreen{1000.0f, 0.0f, 5.0f};
    ok = mip.project(offscreen, 0.1f, level, u, v, cam_z);
    CHECK(!ok);

    // Large r_c → higher mipmap level.
    Vec3f P2{0.0f, 0.0f, 1.0f};
    ok = mip.project(P2, 2.0f, level, u, v, cam_z);
    CHECK(ok);
    CHECK(level >= 1);
}

// ---- CubeHistogram stats ----------------------------------------------------

static void test_histogram_stats() {
    CubeHistogram h;
    // All weight in bin 4 (centre ≈ -1 + 4.5/4 = 0.125).
    h.bins[4] = 10;
    CHECK_NEAR(h.mean_indicator(),   0.125f, 0.01f);
    CHECK_NEAR(h.median_indicator(), 0.125f, 0.01f);

    CubeHistogram h2;
    // Bins 3 and 4 equal → median at bin 3 (centre = -1 + 3.5/4 = -0.125).
    h2.bins[3] = 5;
    h2.bins[4] = 5;
    // Weighted median: cumulative reaches total/2 at bin 3.
    CHECK_NEAR(h2.median_indicator(), -0.125f, 0.01f);

    CubeHistogram empty;
    CHECK_NEAR(empty.mean_indicator(),   0.0f, 1e-6f);
    CHECK_NEAR(empty.median_indicator(), 0.0f, 1e-6f);
}

// ---- compute_histograms correctness -----------------------------------------

static void test_compute_histograms() {
    // Camera at origin, z-forward, simple intrinsics.
    DepthMap dm;
    dm.width = 16; dm.height = 16;
    // Surface at z=5.
    dm.depths.assign(16*16, 5.0f);
    dm.K = Eigen::Matrix3f::Identity();
    dm.K(0,0) = 50; dm.K(1,1) = 50; dm.K(0,2) = 8; dm.K(1,2) = 8;
    dm.Rt = Eigen::Matrix4f::Identity();

    DepthMipmap mip;
    mip.build(dm, false);

    // Scene AABB: [-2, 2]^3 shifted to origin [0, 4]^3, r_root=2, origin=(-2,-2,-2+5-2)
    // Actually let's use origin (0,0,4), r_root=1 so scene z ∈ [4,6].
    Vec3f origin{-0.5f, -0.5f, 4.0f};
    float r_root = 1.0f;

    // One cube at depth 4, cell at scene centre (z≈5, slightly in front of surface).
    // Cell size at d=4: 2/16 = 0.125. Scene centre z = 5.0 (exactly on surface).
    OctreeCube c;
    c.depth = 4;
    c.r_c   = r_root / (1u << 4);  // = 0.0625
    // Cell (4,4,8) at depth 4 → z = 4.0 + (8+0.5)*0.125 = 4.0 + 1.0625 = 5.0625 (just behind surface)
    uint32_t shift = 21 - 4;
    c.code = morton_encode(4u << shift, 4u << shift, 8u << shift);

    std::vector<OctreeCube> cubes = {c};
    std::vector<DepthMipmap> dmaps = {mip};

    auto hists = compute_histograms(cubes, dmaps, origin, r_root);
    CHECK(hists.size() == 1u);
    CHECK(hists[0].total() > 0u);  // should have at least one vote

    // The cube is behind the surface (cam_z > surf_depth would mean a<0).
    // a = surf_depth(5.0) - cam_z(5.0625) = -0.0625 → slightly behind surface.
    // bin = floor((-0.0625/0.375 + 1) * 4) = floor((0.833) * 4) = floor(3.33) = 3
    // So most votes should be in bin 0..3.
    uint32_t front = 0, back = 0;
    for (int k = 0; k < 4; ++k) back  += hists[0].bins[k];
    for (int k = 4; k < 8; ++k) front += hists[0].bins[k];
    CHECK(back > front);  // slightly behind surface

    // ---- save/load roundtrip ----
    std::string path = std::filesystem::temp_directory_path().string() + "/tgv_test_hist.h";
    save_histograms(path, hists);
    auto hists2 = load_histograms(path);
    CHECK(hists2.size() == hists.size());
    for (int k = 0; k < 8; ++k)
        CHECK(hists2[0].bins[k] == hists[0].bins[k]);
    std::filesystem::remove(path);
}

// ---- main -------------------------------------------------------------------

int main() {
    test_mipmap_build();
    test_mipmap_project();
    test_histogram_stats();
    test_compute_histograms();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
