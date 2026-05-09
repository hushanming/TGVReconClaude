#include "octree_builder.hpp"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

int cube_depth_for_radius(float r_x, float r_root, int max_depth) {
    if (r_x <= 0.0f) return max_depth;
    // Find d: 0.75*r_x <= r_root/2^d < 1.5*r_x
    // r_root/2^d >= 0.75*r_x  =>  2^d <= r_root/(0.75*r_x)
    // r_root/2^d < 1.5*r_x   =>  2^d > r_root/(1.5*r_x)
    // Use midpoint: 2^d ~ r_root / (r_x * 1.125)
    int d = static_cast<int>(std::round(std::log2(r_root / (r_x * 1.125f))));
    if (d < 0) d = 0;
    if (d > max_depth) d = max_depth;
    // Verify and nudge if rounding put us outside the valid range.
    for (int tries = 0; tries < 3; ++tries) {
        float cell_size = r_root / static_cast<float>(1u << d);
        if (cell_size >= 0.75f * r_x && cell_size < 1.5f * r_x) break;
        if (cell_size < 0.75f * r_x) { if (d > 0) --d; else break; }
        else                          { if (d < max_depth) ++d; else break; }
    }
    return d;
}

void build_cubes_from_depthmap(
    const DepthMap& dm,
    const Vec3f& scene_origin,
    float r_root,
    const std::string& out_path)
{
    const int W = dm.width, H = dm.height;
    const float inv_2r = 1.0f / (2.0f * r_root);
    const int max_depth = 21;

    // Collect cubes in parallel; protect shared vector with a mutex.
    std::vector<OctreeCube> cubes;
    cubes.reserve(static_cast<size_t>(W) * H / 4);
    std::mutex mtx;

    tbb::parallel_for(tbb::blocked_range<int>(0, H), [&](const tbb::blocked_range<int>& rows) {
        std::vector<OctreeCube> local;
        for (int v = rows.begin(); v < rows.end(); ++v) {
            for (int u = 0; u < W; ++u) {
                if (!dm.valid(u, v)) continue;
                float r_x = dm.sample_radius(u, v);
                if (r_x <= 0.0f) continue;

                Vec3f p = dm.unproject(u, v);
                // Normalize to [0,1]^3 within scene bbox
                Vec3f n = (p - scene_origin) * inv_2r;
                if (n.x() < 0 || n.x() >= 1 || n.y() < 0 || n.y() >= 1 ||
                    n.z() < 0 || n.z() >= 1) continue;

                int d = cube_depth_for_radius(r_x, r_root, max_depth);
                uint32_t grid_size = 1u << d;
                uint32_t ix = static_cast<uint32_t>(n.x() * grid_size);
                uint32_t iy = static_cast<uint32_t>(n.y() * grid_size);
                uint32_t iz = static_cast<uint32_t>(n.z() * grid_size);
                // Clamp to valid range (handles n==1.0 edge case)
                if (ix >= grid_size) ix = grid_size - 1;
                if (iy >= grid_size) iy = grid_size - 1;
                if (iz >= grid_size) iz = grid_size - 1;

                OctreeCube cube;
                int shift = 21 - d;
                cube.code  = morton_encode(ix << shift, iy << shift, iz << shift);
                cube.r_c   = r_x;
                cube.depth = d;
                local.push_back(cube);
            }
        }
        if (!local.empty()) {
            std::lock_guard<std::mutex> lock(mtx);
            cubes.insert(cubes.end(), local.begin(), local.end());
        }
    });

    std::sort(cubes.begin(), cubes.end(), [](const OctreeCube& a, const OctreeCube& b) {
        return a.code < b.code;
    });

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + out_path);

    CubeFileHeader hdr;
    hdr.r_root      = r_root;
    hdr.cube_count  = static_cast<uint32_t>(cubes.size());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!cubes.empty())
        out.write(reinterpret_cast<const char*>(cubes.data()),
                  cubes.size() * sizeof(OctreeCube));
}

std::vector<OctreeCube> load_cubes(const std::string& path, float& r_root_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open cube file: " + path);

    CubeFileHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0] != 'T' || hdr.magic[1] != 'G' ||
        hdr.magic[2] != 'V' || hdr.magic[3] != '1')
        throw std::runtime_error("Bad magic in cube file: " + path);

    r_root_out = hdr.r_root;
    std::vector<OctreeCube> cubes(hdr.cube_count);
    if (hdr.cube_count > 0)
        in.read(reinterpret_cast<char*>(cubes.data()),
                hdr.cube_count * sizeof(OctreeCube));
    return cubes;
}
