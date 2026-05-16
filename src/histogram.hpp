#pragma once
#include "depth_map.hpp"
#include "octree_builder.hpp"
#include <Eigen/Core>
#include <string>
#include <vector>

// 8-bin distance-field histogram for one voxel (Algorithm 1 from paper).
// bin k represents the range a ∈ [-1 + k/4, -1 + (k+1)/4).
struct CubeHistogram {
    uint32_t bins[8] = {};
    uint32_t total() const {
        uint32_t s = 0; for (auto b : bins) s += b; return s;
    }
    // Weighted average of bin centres (rough indicator estimate).
    float mean_indicator() const;
    // Weighted median of bin centres (robust indicator estimate).
    float median_indicator() const;
};

// Depth map with precomputed mipmap pyramid and camera inverse.
struct DepthMipmap {
    std::vector<DepthMap> levels;   // levels[0] = original
    Eigen::Matrix4f Rt_inv;         // cached inverse of Rt (camera←world)
    bool lidar = false;             // vote weight: 5 for LIDAR, 1 for photo

    void build(const DepthMap& base, bool is_lidar = false);

    // Project world point P into the pyramid.
    // Selects the mipmap level whose footprint matches cube radius r_c.
    // Returns false if P is behind the camera or outside image bounds.
    // Sets cam_z to camera-space z-depth of P.
    bool project(const Vec3f& P, float r_c,
                 int& level_out, int& u_out, int& v_out,
                 float& cam_z_out) const;

    float depth_at(int level, int u, int v) const;
};

// Compute 8-bin histograms for every cube in the balanced octree.
// depth_maps: pre-built DepthMipmap instances (photos and/or LIDAR).
// scene_origin: world-space corner of scene AABB.
// r_root: half-size of root cube.
std::vector<CubeHistogram> compute_histograms(
    const std::vector<OctreeCube>& cubes,
    const std::vector<DepthMipmap>& depth_maps,
    const Vec3f& scene_origin,
    float r_root);

// Out-of-core histogram computation: process one treetop leaf at a time.
// Reads cube ranges directly from `balanced_cube_path` (random-access).
// Writes a flat CubeHistogram[total_cubes] file at `out_hist_path`.
// Peak RAM per leaf = N_cubesPerLeaf * (sizeof(OctreeCube) + sizeof(CubeHistogram)).
#include "treetop.hpp"
void compute_histograms_oc(
    const std::string&               balanced_cube_path,
    const std::vector<TreetopLeaf>&  leaves,
    const std::vector<DepthMipmap>&  depth_maps,
    const Vec3f&                     scene_origin,
    float                            r_root,
    const std::string&               out_hist_path);

// File I/O  (magic 'TGVH', version 1, count uint32, then raw CubeHistogram[]).
void save_histograms(const std::string& path,
                     const std::vector<CubeHistogram>& hists);
std::vector<CubeHistogram> load_histograms(const std::string& path);

// Random-access read: load histograms for a range [first, last] (inclusive).
std::vector<CubeHistogram> load_histograms_range(const std::string& path,
                                                  uint32_t first_idx,
                                                  uint32_t last_idx);
