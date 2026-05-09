#include "types.hpp"
#include "octree_builder.hpp"
#include "histogram.hpp"
#include "tgv.hpp"
#include "marching_cubes.hpp"
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

// Build a small 2x2x2 cube arrangement where u flips sign at x=1.
// Cubes at ix=0 have u>0, ix=1 have u<0 → surface between them.
static void test_extract_surface_simple() {
    float r_root = 16.0f;
    int depth = 4;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube> cubes;
    std::vector<TGVCube> state;

    for (uint32_t iz = 0; iz < 2; ++iz)
    for (uint32_t iy = 0; iy < 2; ++iy)
    for (uint32_t ix = 0; ix < 2; ++ix) {
        OctreeCube c;
        c.depth = depth;
        c.r_c   = r_root / (1u << depth);
        c.code  = morton_encode(ix << shift, iy << shift, iz << shift);
        cubes.push_back(c);

        TGVCube s;
        s.u = (ix == 0) ? 1.0f : -1.0f;  // sign flip between ix=0 and ix=1
        state.push_back(s);
    }

    auto tris = extract_surface(cubes, state, Vec3f::Zero(), r_root);

    // There should be triangles: the YZ plane at x=half-cell.
    CHECK(tris.size() > 0u);

    // All triangle vertices should have x ≈ cell_size/2.
    float expected_x = r_root / (1u << depth);  // half cell size (t=0.5 interpolation)
    for (auto& t : tris)
        for (int k = 0; k < 3; ++k)
            CHECK(std::abs(t.v[k].x() - expected_x) < 0.01f * expected_x + 1e-4f);
}

static void test_extract_all_positive() {
    // All u > 0 → no sign changes → no triangles.
    float r_root = 8.0f;
    int depth = 3;
    uint32_t shift = 21 - depth;

    std::vector<OctreeCube> cubes;
    std::vector<TGVCube> state;

    for (uint32_t ix = 0; ix < 2; ++ix) {
        OctreeCube c;
        c.depth = depth;
        c.r_c   = 1.0f;
        c.code  = morton_encode(ix << shift, 0u, 0u);
        cubes.push_back(c);
        TGVCube s; s.u = 1.0f;
        state.push_back(s);
    }

    auto tris = extract_surface(cubes, state, Vec3f::Zero(), r_root);
    CHECK(tris.empty());
}

static void test_write_obj() {
    // Create a trivial triangle and write it.
    std::vector<Triangle> tris;
    Triangle t;
    t.v[0] = Vec3f{0, 0, 0};
    t.v[1] = Vec3f{1, 0, 0};
    t.v[2] = Vec3f{0, 1, 0};
    tris.push_back(t);

    std::string path = std::filesystem::temp_directory_path().string() + "/tgv_test_mesh.obj";
    write_obj(path, tris);

    // Verify file exists and has non-trivial size.
    CHECK(std::filesystem::file_size(path) > 10u);
    std::filesystem::remove(path);
}

int main() {
    test_extract_surface_simple();
    test_extract_all_positive();
    test_write_obj();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
