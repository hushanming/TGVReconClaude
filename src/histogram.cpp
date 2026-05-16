#include "histogram.hpp"
#include "octree_builder.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

// ---------- CubeHistogram ----------------------------------------------------

float CubeHistogram::mean_indicator() const {
    uint32_t tot = total();
    if (tot == 0) return 0.0f;
    float sum = 0.0f;
    for (int k = 0; k < 8; ++k)
        sum += bins[k] * (-1.0f + (k + 0.5f) / 4.0f);
    return sum / tot;
}

float CubeHistogram::median_indicator() const {
    uint32_t tot = total();
    if (tot == 0) return 0.0f;
    uint32_t cum = 0;
    for (int k = 0; k < 8; ++k) {
        cum += bins[k];
        if (cum * 2 >= tot)
            return -1.0f + (k + 0.5f) / 4.0f;
    }
    return 1.0f;
}

// ---------- DepthMipmap ------------------------------------------------------

void DepthMipmap::build(const DepthMap& base, bool is_lidar) {
    lidar = is_lidar;
    Rt_inv = base.Rt.inverse();

    levels.clear();
    levels.push_back(base);

    for (int l = 0; l < 12; ++l) {
        const DepthMap& prev = levels.back();
        if (prev.width <= 1 || prev.height <= 1) break;

        DepthMap next;
        next.width  = std::max(1, prev.width  / 2);
        next.height = std::max(1, prev.height / 2);
        next.K  = prev.K;
        next.K(0,0) *= 0.5f; next.K(1,1) *= 0.5f;
        next.K(0,2)  = prev.K(0,2) * 0.5f;
        next.K(1,2)  = prev.K(1,2) * 0.5f;
        next.Rt = prev.Rt;

        const float nan = std::numeric_limits<float>::quiet_NaN();
        next.depths.assign(next.width * next.height, nan);

        for (int v = 0; v < next.height; ++v)
            for (int u = 0; u < next.width; ++u) {
                float sum = 0.0f; int cnt = 0;
                for (int dv = 0; dv < 2; ++dv)
                    for (int du = 0; du < 2; ++du) {
                        int pu = 2*u+du, pv = 2*v+dv;
                        if (pu < prev.width && pv < prev.height) {
                            float d = prev.depths[pv * prev.width + pu];
                            if (std::isfinite(d) && d > 0.0f) { sum += d; ++cnt; }
                        }
                    }
                if (cnt > 0)
                    next.depths[v * next.width + u] = sum / cnt;
            }
        levels.push_back(std::move(next));
    }
}

bool DepthMipmap::project(const Vec3f& P, float r_c,
                           int& level_out, int& u_out, int& v_out,
                           float& cam_z_out) const {
    if (levels.empty()) return false;
    const DepthMap& base = levels[0];

    Eigen::Vector4f Pw{P.x(), P.y(), P.z(), 1.0f};
    Eigen::Vector4f cam = Rt_inv * Pw;
    float z = cam.z();
    if (z <= 0.0f) return false;

    float fx = base.K(0,0), fy = base.K(1,1), cx = base.K(0,2), cy = base.K(1,2);
    float u_f = cam.x() / z * fx + cx;
    float v_f = cam.y() / z * fy + cy;

    int u0 = static_cast<int>(u_f);
    int v0 = static_cast<int>(v_f);
    if (u0 < 0 || u0 >= base.width || v0 < 0 || v0 >= base.height) return false;

    cam_z_out = z;

    // Mipmap level: choose so that pixel footprint ≈ r_c * fx / z pixels.
    float footprint = std::abs(fx) * r_c / z;
    int level = std::max(0, static_cast<int>(std::log2(std::max(1.0f, footprint))));
    level = std::min(level, static_cast<int>(levels.size()) - 1);

    level_out = level;
    u_out = std::clamp(u0 >> level, 0, levels[level].width  - 1);
    v_out = std::clamp(v0 >> level, 0, levels[level].height - 1);
    return true;
}

float DepthMipmap::depth_at(int level, int u, int v) const {
    if (level < 0 || level >= (int)levels.size()) return std::numeric_limits<float>::quiet_NaN();
    const DepthMap& lm = levels[level];
    if (u < 0 || u >= lm.width || v < 0 || v >= lm.height) return std::numeric_limits<float>::quiet_NaN();
    return lm.depths[v * lm.width + u];
}

// ---------- histogram computation -------------------------------------------

// World-space center of an OctreeCube (fine-grid Morton code convention).
static Vec3f cube_center(const OctreeCube& c, const Vec3f& origin, float r_root) {
    uint32_t fx, fy, fz;
    morton_decode(c.code, fx, fy, fz);
    int shift = 21 - c.depth;
    uint32_t ix = fx >> shift, iy = fy >> shift, iz = fz >> shift;
    float cs = 2.0f * r_root / static_cast<float>(1u << c.depth);
    return origin + Vec3f{(ix + 0.5f) * cs, (iy + 0.5f) * cs, (iz + 0.5f) * cs};
}

std::vector<CubeHistogram> compute_histograms(
    const std::vector<OctreeCube>& cubes,
    const std::vector<DepthMipmap>& depth_maps,
    const Vec3f& origin,
    float r_root)
{
    std::vector<CubeHistogram> hists(cubes.size());

    for (size_t ci = 0; ci < cubes.size(); ++ci) {
        const OctreeCube& cube = cubes[ci];
        Vec3f center = cube_center(cube, origin, r_root);
        float r_c   = cube.r_c;
        float delta = 6.0f * r_c;   // δ_x = 6·r_x  (near-surface width)
        float eta   = 18.0f * r_c;  // η_x = 18·r_x (occluded region width)
        if (r_c <= 0.0f) continue;

        for (const auto& dm : depth_maps) {
            int level, u_mip, v_mip;
            float cam_z;
            if (!dm.project(center, r_c, level, u_mip, v_mip, cam_z)) continue;

            float surf_depth = dm.depth_at(level, u_mip, v_mip);
            if (!std::isfinite(surf_depth) || surf_depth <= 0.0f) continue;

            float a = surf_depth - cam_z;
            if (a < -eta) continue;  // behind occluded region: skip

            a = std::clamp(a / delta, -1.0f, 1.0f);
            int bin = static_cast<int>((a + 1.0f) * 4.0f);  // map [-1,1] → [0,8)
            bin = std::clamp(bin, 0, 7);

            hists[ci].bins[bin] += dm.lidar ? 5u : 1u;
        }
    }

    return hists;
}

// ---------- file I/O ---------------------------------------------------------

namespace {
struct HistFileHeader {
    char     magic[4] = {'T', 'G', 'V', 'H'};
    uint32_t version  = 1;
    uint32_t count    = 0;
};
}

void save_histograms(const std::string& path,
                     const std::vector<CubeHistogram>& hists) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open: " + path);
    HistFileHeader hdr;
    hdr.count = static_cast<uint32_t>(hists.size());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!hists.empty())
        out.write(reinterpret_cast<const char*>(hists.data()),
                  hists.size() * sizeof(CubeHistogram));
}

std::vector<CubeHistogram> load_histograms(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    HistFileHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='H')
        throw std::runtime_error("Bad histogram magic");
    std::vector<CubeHistogram> hists(hdr.count);
    if (hdr.count > 0)
        in.read(reinterpret_cast<char*>(hists.data()),
                hdr.count * sizeof(CubeHistogram));
    return hists;
}

std::vector<CubeHistogram> load_histograms_range(const std::string& path,
                                                  uint32_t first_idx,
                                                  uint32_t last_idx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    HistFileHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='H')
        throw std::runtime_error("Bad histogram magic");
    if (first_idx >= hdr.count || first_idx > last_idx) return {};
    uint32_t end   = std::min(last_idx + 1, hdr.count);
    uint32_t count = end - first_idx;
    in.seekg(sizeof(HistFileHeader) +
             static_cast<std::streamoff>(first_idx) * sizeof(CubeHistogram));
    std::vector<CubeHistogram> hists(count);
    in.read(reinterpret_cast<char*>(hists.data()), count * sizeof(CubeHistogram));
    return hists;
}

// ---------- out-of-core histogram computation --------------------------------

void compute_histograms_oc(
    const std::string&               balanced_cube_path,
    const std::vector<TreetopLeaf>&  leaves,
    const std::vector<DepthMipmap>&  depth_maps,
    const Vec3f&                     scene_origin,
    float                            r_root,
    const std::string&               out_hist_path)
{
    // Count total cubes from last leaf.
    if (leaves.empty()) {
        std::ofstream out(out_hist_path, std::ios::binary);
        HistFileHeader hdr; hdr.count = 0;
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        return;
    }
    uint32_t total_cubes = leaves.back().last_idx + 1;

    // Write header, then stream leaf histograms in index order.
    std::ofstream out(out_hist_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open: " + out_hist_path);
    HistFileHeader hdr; hdr.count = total_cubes;
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // Leaves are already sorted by first_idx (they span the entire cube file).
    float r_root_tmp;
    for (const auto& leaf : leaves) {
        auto leaf_cubes = load_cubes_range(balanced_cube_path,
                                           leaf.first_idx, leaf.last_idx,
                                           r_root_tmp);
        auto leaf_hists = compute_histograms(leaf_cubes, depth_maps,
                                             scene_origin, r_root);
        out.write(reinterpret_cast<const char*>(leaf_hists.data()),
                  leaf_hists.size() * sizeof(CubeHistogram));
    }
}
