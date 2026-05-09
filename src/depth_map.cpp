#include "depth_map.hpp"
#include <Eigen/Geometry>

Vec3f DepthMap::unproject(int u, int v) const {
    float d = depths[v * width + u];
    // Camera-space point: K_inv * [u, v, 1] * d
    float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    Eigen::Vector4f cam{(u - cx) * d / fx, (v - cy) * d / fy, d, 1.0f};
    Eigen::Vector4f world = Rt * cam;
    return world.head<3>();
}

float DepthMap::sample_radius(int u, int v) const {
    if (!valid(u, v)) return 0.0f;
    Vec3f center = unproject(u, v);

    const int du[] = {1, -1, 0, 0};
    const int dv[] = {0,  0, 1, -1};

    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        int nu = u + du[i], nv = v + dv[i];
        if (!valid(nu, nv)) continue;
        sum += (unproject(nu, nv) - center).norm();
        ++count;
    }
    if (count == 0) return 0.0f;
    return 0.5f * sum / count;
}
