#pragma once
#include "types.hpp"
#include <Eigen/Core>
#include <cmath>
#include <vector>

struct DepthMap {
    int width = 0, height = 0;
    std::vector<float> depths;   // row-major; NaN = invalid pixel
    Eigen::Matrix3f K;           // camera intrinsics
    Eigen::Matrix4f Rt;          // world←camera transform (4x4 homogeneous)

    bool valid(int u, int v) const {
        if (u < 0 || v < 0 || u >= width || v >= height) return false;
        float d = depths[v * width + u];
        return std::isfinite(d) && d > 0.0f;
    }

    // Unproject pixel (u,v) with stored depth to world-space 3D point.
    Vec3f unproject(int u, int v) const;

    // Estimate r_x: half of mean 3D distance to valid 4-connected neighbors.
    // Returns 0 if no valid neighbors exist.
    float sample_radius(int u, int v) const;
};
