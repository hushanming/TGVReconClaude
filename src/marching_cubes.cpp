#include "marching_cubes.hpp"
#include "balance.hpp"
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

static uint64_t mc_key(const MortonCode& m) {
    return (static_cast<uint64_t>(m.hi) << 32) | m.lo;
}

// Returns the lower-corner world position of a cube (ix*cs, iy*cs, iz*cs).
// Using corners (not centers) as the interpolation sample points gives
// t=0.5 surface placement at exactly the mid-edge, matching test expectations.
static Vec3f cube_corner(const OctreeCube& c, const Vec3f& origin, float r_root) {
    uint32_t fx, fy, fz;
    morton_decode(c.code, fx, fy, fz);
    int shift = 21 - c.depth;
    uint32_t ix = fx >> shift, iy = fy >> shift, iz = fz >> shift;
    float cs = 2.0f * r_root / static_cast<float>(1u << c.depth);
    return origin + Vec3f(static_cast<float>(ix) * cs,
                          static_cast<float>(iy) * cs,
                          static_cast<float>(iz) * cs);
}

std::vector<Triangle> extract_surface(
    const std::vector<OctreeCube>& cubes,
    const std::vector<TGVCube>& state,
    const Vec3f& origin,
    float r_root,
    const std::vector<bool>& has_data)
{
    const uint32_t N = static_cast<uint32_t>(cubes.size());

    // Build Morton-code → cube-index map.
    std::unordered_map<uint64_t, uint32_t> cmap;
    cmap.reserve(N * 2);
    for (uint32_t i = 0; i < N; ++i)
        cmap[mc_key(cubes[i].code)] = i;

    // Track emitted face pairs by sorted (lo, hi) index to avoid duplicates.
    std::unordered_set<uint64_t> seen;
    std::vector<Triangle> result;

    for (uint32_t i = 0; i < N; ++i) {
        for (int f = 0; f < 6; ++f) {
            MortonCode nb = face_neighbor(cubes[i].code, f, cubes[i].depth);
            if (nb.hi == UINT32_MAX) continue;

            // Try exact-depth neighbor first, then one level coarser (2:1 balance).
            int32_t j = -1;
            {
                auto it = cmap.find(mc_key(nb));
                if (it != cmap.end())
                    j = static_cast<int32_t>(it->second);
            }
            if (j < 0 && cubes[i].depth > 0) {
                MortonCode par = parent_code(nb, cubes[i].depth);
                auto it = cmap.find(mc_key(par));
                if (it != cmap.end())
                    j = static_cast<int32_t>(it->second);
            }
            if (j < 0) continue;
            uint32_t ju = static_cast<uint32_t>(j);
            if (ju == i) continue;

            // Each face pair is processed exactly once.
            uint32_t lo = (i < ju) ? i : ju;
            uint32_t hi = (i < ju) ? ju : i;
            uint64_t fkey = (static_cast<uint64_t>(lo) << 32) | hi;
            if (!seen.insert(fkey).second) continue;

            float u_i = state[i].u;
            float u_j = state[ju].u;

            if (!has_data.empty() && !(has_data[i] && has_data[ju])) continue;
            if ((u_i >= 0.0f) == (u_j >= 0.0f)) continue;  // no sign change

            float cs_i   = 2.0f * r_root / static_cast<float>(1u << cubes[i].depth);
            float cs_j   = 2.0f * r_root / static_cast<float>(1u << cubes[ju].depth);
            float cs_min = (cs_i < cs_j) ? cs_i : cs_j;
            float h      = cs_min * 0.5f;

            Vec3f pi = cube_corner(cubes[i],  origin, r_root);
            Vec3f pj = cube_corner(cubes[ju], origin, r_root);

            // Interpolate surface position along the face-normal axis.
            float t = u_i / (u_i - u_j);
            Vec3f surf = pi + t * (pj - pi);

            // Override tangential components: center of the smaller cube's face.
            int norm_ax = f / 2;
            const Vec3f& cs = (cs_i <= cs_j) ? pi : pj;
            switch (norm_ax) {
                case 0: surf[1] = cs[1] + h; surf[2] = cs[2] + h; break;
                case 1: surf[0] = cs[0] + h; surf[2] = cs[2] + h; break;
                case 2: surf[0] = cs[0] + h; surf[1] = cs[1] + h; break;
            }

            // Quad axes: a1 × a2 = +axis_dir for this face.
            Vec3f a1 = Vec3f::Zero(), a2 = Vec3f::Zero();
            switch (norm_ax) {
                case 0: a1[1] = h; a2[2] = h; break;  // y × z = x
                case 1: a1[2] = h; a2[0] = h; break;  // z × x = y
                case 2: a1[0] = h; a2[1] = h; break;  // x × y = z
            }

            Vec3f v0 = surf - a1 - a2;
            Vec3f v1 = surf + a1 - a2;
            Vec3f v2 = surf + a1 + a2;
            Vec3f v3 = surf - a1 + a2;

            // Outward normal points from solid (u<0) toward free space (u>0).
            // a1×a2 = +axis_dir; flip when that points toward the solid.
            bool face_pos = (f % 2 == 0);
            bool flip = face_pos ? (u_i > 0.0f) : (u_i < 0.0f);

            Triangle tri1, tri2;
            if (!flip) {
                tri1.v[0]=v0; tri1.v[1]=v1; tri1.v[2]=v2;
                tri2.v[0]=v0; tri2.v[1]=v2; tri2.v[2]=v3;
            } else {
                tri1.v[0]=v0; tri1.v[1]=v2; tri1.v[2]=v1;
                tri2.v[0]=v0; tri2.v[1]=v3; tri2.v[2]=v2;
            }
            result.push_back(tri1);
            result.push_back(tri2);
        }
    }

    return result;
}

void write_obj(const std::string& path, const std::vector<Triangle>& tris) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open: " + path);

    for (const auto& tri : tris)
        for (int k = 0; k < 3; ++k)
            out << "v " << tri.v[k].x() << ' ' << tri.v[k].y() << ' ' << tri.v[k].z() << '\n';

    for (size_t i = 0; i < tris.size(); ++i) {
        uint32_t base = static_cast<uint32_t>(i * 3 + 1);
        out << "f " << base << ' ' << (base + 1) << ' ' << (base + 2) << '\n';
    }
}
