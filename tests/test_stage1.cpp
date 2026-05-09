#include "types.hpp"
#include "depth_map.hpp"
#include "octree_builder.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Minimal test framework
static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)
#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a)-(b)) <= (eps))

// --- Morton encode/decode roundtrip ---
static void test_morton_roundtrip() {
    for (uint32_t ix : {0u, 1u, 127u, 1023u, 2097151u}) {
        for (uint32_t iy : {0u, 5u, 255u}) {
            for (uint32_t iz : {0u, 3u, 100u}) {
                MortonCode m = morton_encode(ix, iy, iz);
                uint32_t dx, dy, dz;
                morton_decode(m, dx, dy, dz);
                CHECK(dx == ix);
                CHECK(dy == iy);
                CHECK(dz == iz);
            }
        }
    }
}

// --- Morton ordering is a Z-curve (all x-axis increments stay monotone when y,z fixed) ---
static void test_morton_ordering() {
    for (uint32_t i = 0; i + 1 < 100; ++i) {
        MortonCode a = morton_encode(i, 7, 3);
        MortonCode b = morton_encode(i + 1, 7, 3);
        CHECK(a < b);
    }
}

// --- cube_depth_for_radius: verify Eq.4 constraint ---
static void test_cube_depth() {
    float r_root = 100.0f;
    for (float r_x : {0.1f, 1.0f, 5.0f, 10.0f, 50.0f}) {
        int d = cube_depth_for_radius(r_x, r_root);
        float cell = r_root / static_cast<float>(1u << d);
        // 0.75*r_x <= cell < 1.5*r_x
        CHECK(cell >= 0.75f * r_x);
        CHECK(cell <  1.5f  * r_x);
    }
}

// --- DepthMap::unproject and sample_radius ---
static void test_depth_map_unproject() {
    // Synthetic depth map: 5x5 plane at z=10, centered at (0,0,10)
    // K: fx=fy=100, cx=2, cy=2
    DepthMap dm;
    dm.width = 5; dm.height = 5;
    dm.depths.assign(25, 10.0f);
    dm.K = Eigen::Matrix3f::Identity();
    dm.K(0,0) = 100; dm.K(1,1) = 100; dm.K(0,2) = 2; dm.K(1,2) = 2;
    dm.Rt = Eigen::Matrix4f::Identity(); // camera at origin, z-forward

    // Pixel (2,2) should unproject to (0, 0, 10)
    Vec3f p = dm.unproject(2, 2);
    CHECK_NEAR(p.x(), 0.0f, 1e-4f);
    CHECK_NEAR(p.y(), 0.0f, 1e-4f);
    CHECK_NEAR(p.z(), 10.0f, 1e-4f);

    // Pixel (3,2) should be (0.1, 0, 10) since (3-2)/100 * 10 = 0.1
    p = dm.unproject(3, 2);
    CHECK_NEAR(p.x(), 0.1f, 1e-4f);

    // sample_radius at center (2,2): neighbors at x+-0.1, y+-0.1 in world.
    // dist to each ~= 0.1 => r_x = 0.5 * 0.1 = 0.05
    float r = dm.sample_radius(2, 2);
    CHECK_NEAR(r, 0.05f, 1e-4f);
}

// --- build_cubes_from_depthmap + load_cubes roundtrip ---
static void test_build_and_load() {
    DepthMap dm;
    dm.width = 10; dm.height = 10;
    dm.depths.assign(100, 5.0f);
    dm.K = Eigen::Matrix3f::Identity();
    dm.K(0,0) = 50; dm.K(1,1) = 50; dm.K(0,2) = 5; dm.K(1,2) = 5;
    dm.Rt = Eigen::Matrix4f::Identity();

    // World points span roughly x in [-0.9, 0.9], y in [-0.9, 0.9], z=5
    // Place scene origin at (-1, -1, 4), r_root=2.5 so all points fall in [0,1]^3
    Vec3f origin(-1.0f, -1.0f, 4.0f);
    float r_root = 2.5f;

    std::string tmp = std::filesystem::temp_directory_path().string() + "/tgv_test.cubes";
    build_cubes_from_depthmap(dm, origin, r_root, tmp);

    float r_root_loaded;
    auto cubes = load_cubes(tmp, r_root_loaded);
    CHECK_NEAR(r_root_loaded, r_root, 1e-5f);
    CHECK(!cubes.empty());

    // All cubes sorted
    for (size_t i = 1; i < cubes.size(); ++i)
        CHECK(!(cubes[i].code < cubes[i-1].code));

    // All cube depths satisfy Eq.4
    for (const auto& c : cubes) {
        float cell = r_root / static_cast<float>(1u << c.depth);
        CHECK(cell >= 0.75f * c.r_c);
        CHECK(cell <  1.5f  * c.r_c);
    }

    std::filesystem::remove(tmp);
}

int main() {
    test_morton_roundtrip();
    test_morton_ordering();
    test_cube_depth();
    test_depth_map_unproject();
    test_build_and_load();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
