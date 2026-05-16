#include "marching_cubes.hpp"
#include "balance.hpp"
#include "histogram.hpp"
#include "treetop.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------- helpers ----------------------------------------------------------

static uint64_t mc_key(const MortonCode& m) {
    return (static_cast<uint64_t>(m.hi) << 32) | m.lo;
}

// World-space lower-corner of a cube.
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

// Quantise a world-space coordinate to a 20-bit integer index.
// Resolution = 2*r_root / 2^20 ≈ 1.9e-6 of scene size.
static uint32_t quantise(float v, float origin, float inv_scale) {
    int q = static_cast<int>((v - origin) * inv_scale + 0.5f);
    if (q < 0) q = 0;
    if (q > (1 << 20) - 1) q = (1 << 20) - 1;
    return static_cast<uint32_t>(q);
}

// 60-bit vertex key packed as qx[19:0] | qy[39:20] | qz[59:40].
static uint64_t vertex_key(const Vec3f& p, const Vec3f& origin, float inv_scale) {
    uint64_t qx = quantise(p.x(), origin.x(), inv_scale);
    uint64_t qy = quantise(p.y(), origin.y(), inv_scale);
    uint64_t qz = quantise(p.z(), origin.z(), inv_scale);
    return qx | (qy << 20) | (qz << 40);
}

// Look up or insert a vertex; return its index.
static uint32_t get_or_insert(
    std::unordered_map<uint64_t, uint32_t>& vmap,
    std::vector<Vec3f>& verts,
    const Vec3f& pos, const Vec3f& origin, float inv_scale)
{
    uint64_t k = vertex_key(pos, origin, inv_scale);
    auto [it, inserted] = vmap.emplace(k, static_cast<uint32_t>(verts.size()));
    if (inserted) verts.push_back(pos);
    return it->second;
}

// ---------- surface extraction -----------------------------------------------

Mesh extract_surface(
    const std::vector<OctreeCube>& cubes,
    const std::vector<TGVCube>&    state,
    const Vec3f&                   origin,
    float                          r_root,
    const std::vector<bool>&       has_data)
{
    const uint32_t N = static_cast<uint32_t>(cubes.size());
    const float inv_scale = static_cast<float>(1u << 20) / (2.0f * r_root);

    // Build Morton-code → cube-index lookup.
    std::unordered_map<uint64_t, uint32_t> cmap;
    cmap.reserve(N * 2);
    for (uint32_t i = 0; i < N; ++i)
        cmap[mc_key(cubes[i].code)] = i;

    // Per-face-pair deduplication (sorted index pair → already emitted).
    std::unordered_set<uint64_t> seen;

    // Vertex deduplication map and output arrays.
    std::unordered_map<uint64_t, uint32_t> vmap;
    vmap.reserve(N * 3);
    Mesh mesh;
    mesh.verts.reserve(N * 2);
    mesh.faces.reserve(N * 4);

    for (uint32_t i = 0; i < N; ++i) {
        for (int f = 0; f < 6; ++f) {
            MortonCode nb = face_neighbor(cubes[i].code, f, cubes[i].depth);
            if (nb.hi == UINT32_MAX) continue;

            // Try exact-depth neighbor, then one level coarser (2:1 balance).
            int32_t j = -1;
            {
                auto it = cmap.find(mc_key(nb));
                if (it != cmap.end()) j = static_cast<int32_t>(it->second);
            }
            if (j < 0 && cubes[i].depth > 0) {
                MortonCode par = parent_code(nb, cubes[i].depth);
                auto it = cmap.find(mc_key(par));
                if (it != cmap.end()) j = static_cast<int32_t>(it->second);
            }
            if (j < 0) continue;
            uint32_t ju = static_cast<uint32_t>(j);
            if (ju == i) continue;

            // Each face pair emitted exactly once.
            uint32_t lo = (i < ju) ? i : ju;
            uint32_t hi = (i < ju) ? ju : i;
            uint64_t fkey = (static_cast<uint64_t>(lo) << 32) | hi;
            if (!seen.insert(fkey).second) continue;

            float u_i = state[i].u;
            float u_j = state[ju].u;

            if (!has_data.empty() && !(has_data[i] && has_data[ju])) continue;
            if ((u_i >= 0.0f) == (u_j >= 0.0f)) continue; // no sign change

            float cs_i   = 2.0f * r_root / static_cast<float>(1u << cubes[i].depth);
            float cs_j   = 2.0f * r_root / static_cast<float>(1u << cubes[ju].depth);
            float cs_min = (cs_i < cs_j) ? cs_i : cs_j;
            float h      = cs_min * 0.5f;

            Vec3f pi = cube_corner(cubes[i],  origin, r_root);
            Vec3f pj = cube_corner(cubes[ju], origin, r_root);

            // Zero-crossing: interpolate between cube CENTRES (not lower corners).
            // ci_norm = pi[norm_ax] + cs_i/2, cj_norm = pj[norm_ax] + cs_j/2.
            float t = u_i / (u_i - u_j);
            int norm_ax = f / 2;
            Vec3f surf;
            surf[norm_ax] = (pi[norm_ax] + cs_i * 0.5f)
                          + t * ((pj[norm_ax] + cs_j * 0.5f)
                               - (pi[norm_ax] + cs_i * 0.5f));

            // Tangential components: centre of the smaller cube's face.
            const Vec3f& small_corner = (cs_i <= cs_j) ? pi : pj;
            switch (norm_ax) {
                case 0: surf[1] = small_corner[1] + h; surf[2] = small_corner[2] + h; break;
                case 1: surf[0] = small_corner[0] + h; surf[2] = small_corner[2] + h; break;
                case 2: surf[0] = small_corner[0] + h; surf[1] = small_corner[1] + h; break;
            }

            // Quad half-extents along the two tangential axes.
            Vec3f a1 = Vec3f::Zero(), a2 = Vec3f::Zero();
            switch (norm_ax) {
                case 0: a1[1] = h; a2[2] = h; break; // y×z → x normal
                case 1: a1[2] = h; a2[0] = h; break; // z×x → y normal
                case 2: a1[0] = h; a2[1] = h; break; // x×y → z normal
            }

            // Quad corners (shared with adjacent quads via vertex map).
            Vec3f v0 = surf - a1 - a2;
            Vec3f v1 = surf + a1 - a2;
            Vec3f v2 = surf + a1 + a2;
            Vec3f v3 = surf - a1 + a2;

            uint32_t i0 = get_or_insert(vmap, mesh.verts, v0, origin, inv_scale);
            uint32_t i1 = get_or_insert(vmap, mesh.verts, v1, origin, inv_scale);
            uint32_t i2 = get_or_insert(vmap, mesh.verts, v2, origin, inv_scale);
            uint32_t i3 = get_or_insert(vmap, mesh.verts, v3, origin, inv_scale);

            // Winding: outward normal points from solid (u<0) toward free (u>0).
            // a1×a2 = +axis_dir; flip when that points into the solid.
            bool face_pos = (f % 2 == 0);
            bool flip     = face_pos ? (u_i > 0.0f) : (u_i < 0.0f);

            if (!flip) {
                mesh.faces.push_back(i0); mesh.faces.push_back(i1); mesh.faces.push_back(i2);
                mesh.faces.push_back(i0); mesh.faces.push_back(i2); mesh.faces.push_back(i3);
            } else {
                mesh.faces.push_back(i0); mesh.faces.push_back(i2); mesh.faces.push_back(i1);
                mesh.faces.push_back(i0); mesh.faces.push_back(i3); mesh.faces.push_back(i2);
            }
        }
    }

    return mesh;
}

// ---------- normal computation -----------------------------------------------

void compute_normals(Mesh& mesh) {
    const size_t nv = mesh.verts.size();
    const size_t nt = mesh.tri_count();
    mesh.normals.assign(nv, Vec3f::Zero());

    for (size_t t = 0; t < nt; ++t) {
        uint32_t a = mesh.faces[t * 3];
        uint32_t b = mesh.faces[t * 3 + 1];
        uint32_t c = mesh.faces[t * 3 + 2];
        Vec3f n = (mesh.verts[b] - mesh.verts[a])
                       .cross(mesh.verts[c] - mesh.verts[a]);
        mesh.normals[a] += n;
        mesh.normals[b] += n;
        mesh.normals[c] += n;
    }

    for (auto& n : mesh.normals) {
        float len = n.norm();
        if (len > 1e-12f) n /= len;
    }
}

// ---------- streaming surface extraction -------------------------------------
//
// Each leaf is processed independently. A face pair (i, j) is emitted by the
// "owner" leaf, where ownership is:
//   - cubes[i].depth >  cubes[j].depth  → owned by the finer side (only finer
//     side ever sees the CTF parent neighbour);
//   - cubes[i].depth == cubes[j].depth  → owned by the side with smaller global
//     index (equal-depth pairs are visible from both sides).
//
// Each leaf writes its quads to a temporary binary file as fixed-size records:
//   struct { uint64_t key[4]; uint8_t flip; }  (8-byte aligned, 40 B/quad).
// A 60-bit vertex key encodes (qx,qy,qz) — same quantisation used by the
// in-RAM extract_surface, so identical positions across leaf boundaries
// collide and the final mesh is seamless.
//
// The aggregator passes:
//   pass 1 → collect unique keys, assign global vertex indices, write `v` lines;
//   pass 2 → translate each quad's keys into face indices, write triangle `f` lines.

namespace {

struct QuadRec {
    uint64_t key[4];
    uint8_t  flip;
    uint8_t  _pad[7];   // round to 40 B
};
static_assert(sizeof(QuadRec) == 40, "QuadRec layout");

// Random-access cube file reader with binary search by Morton code.
struct CubeReader {
    std::ifstream    in;
    CubeFileHeader   hdr;

    explicit CubeReader(const std::string& path) : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("Cannot open: " + path);
        in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='1')
            throw std::runtime_error("Bad cube file: " + path);
    }
    uint32_t count() const { return hdr.cube_count; }

    OctreeCube read_at(uint32_t idx) {
        in.seekg(sizeof(CubeFileHeader) +
                 static_cast<std::streamoff>(idx) * sizeof(OctreeCube));
        OctreeCube c;
        in.read(reinterpret_cast<char*>(&c), sizeof(c));
        return c;
    }
    uint32_t lookup(const MortonCode& target) {
        uint32_t lo = 0, hi = hdr.cube_count;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            OctreeCube c = read_at(mid);
            if (c.code == target) return mid;
            if (c.code < target) lo = mid + 1;
            else                 hi = mid;
        }
        return UINT32_MAX;
    }
};

// Random-access TGV state and hist readers (header sizes fixed).
struct TgvStateReader {
    std::ifstream in;
    uint32_t      count = 0;
    explicit TgvStateReader(const std::string& path) : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("Cannot open: " + path);
        char m[4]; uint32_t ver, cnt;
        in.read(m, 4); in.read(reinterpret_cast<char*>(&ver), 4);
        in.read(reinterpret_cast<char*>(&cnt), 4);
        count = cnt;
    }
    TGVCube read_at(uint32_t idx) {
        in.seekg(12 + static_cast<std::streamoff>(idx) * sizeof(TGVCube));
        TGVCube s;
        in.read(reinterpret_cast<char*>(&s), sizeof(s));
        return s;
    }
};
struct HistReader {
    std::ifstream in;
    uint32_t      count = 0;
    explicit HistReader(const std::string& path) : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("Cannot open: " + path);
        char m[4]; uint32_t ver, cnt;
        in.read(m, 4); in.read(reinterpret_cast<char*>(&ver), 4);
        in.read(reinterpret_cast<char*>(&cnt), 4);
        count = cnt;
    }
    CubeHistogram read_at(uint32_t idx) {
        in.seekg(12 + static_cast<std::streamoff>(idx) * sizeof(CubeHistogram));
        CubeHistogram h;
        in.read(reinterpret_cast<char*>(&h), sizeof(h));
        return h;
    }
};

uint64_t mc_key2(const MortonCode& m) {
    return (static_cast<uint64_t>(m.hi) << 32) | m.lo;
}

Vec3f cube_corner2(const OctreeCube& c, const Vec3f& origin, float r_root) {
    uint32_t fx, fy, fz;
    morton_decode(c.code, fx, fy, fz);
    int shift = 21 - c.depth;
    uint32_t ix = fx >> shift, iy = fy >> shift, iz = fz >> shift;
    float cs = 2.0f * r_root / static_cast<float>(1u << c.depth);
    return origin + Vec3f(ix * cs, iy * cs, iz * cs);
}

uint32_t quantise2(float v, float origin, float inv_scale) {
    int q = static_cast<int>((v - origin) * inv_scale + 0.5f);
    if (q < 0) q = 0;
    if (q > (1 << 20) - 1) q = (1 << 20) - 1;
    return static_cast<uint32_t>(q);
}
uint64_t vkey(const Vec3f& p, const Vec3f& origin, float inv_scale) {
    uint64_t qx = quantise2(p.x(), origin.x(), inv_scale);
    uint64_t qy = quantise2(p.y(), origin.y(), inv_scale);
    uint64_t qz = quantise2(p.z(), origin.z(), inv_scale);
    return qx | (qy << 20) | (qz << 40);
}
Vec3f dequantise(uint64_t k, const Vec3f& origin, float scale) {
    uint64_t qx = k & 0xFFFFFull;
    uint64_t qy = (k >> 20) & 0xFFFFFull;
    uint64_t qz = (k >> 40) & 0xFFFFFull;
    return origin + Vec3f(qx * scale, qy * scale, qz * scale);
}

} // namespace

void extract_surface_streaming(
    const std::string& balanced_cube_path,
    const std::string& tgv_state_path,
    const std::string& treetop_path,
    const std::string& hist_path,
    const Vec3f&       origin,
    float              r_root,
    const std::string& out_obj_path)
{
    const float inv_scale = static_cast<float>(1u << 20) / (2.0f * r_root);
    const float scale     = 1.0f / inv_scale;

    float r_root_tt;
    auto leaves = load_treetop(treetop_path, r_root_tt);

    std::string quads_path = out_obj_path + ".quads.tmp";
    std::ofstream qout(quads_path, std::ios::binary | std::ios::trunc);
    if (!qout) throw std::runtime_error("Cannot open: " + quads_path);

    // -------------------------------------------------------------------------
    // Pass A: per-leaf extraction → quads file
    // -------------------------------------------------------------------------
    for (const auto& leaf : leaves) {
        auto cubes_leaf = load_cubes_range(balanced_cube_path,
                                            leaf.first_idx, leaf.last_idx, r_root_tt);
        if (cubes_leaf.empty()) continue;
        auto hists_leaf = load_histograms_range(hist_path,
                                                 leaf.first_idx, leaf.last_idx);
        // State for the leaf range (in-leaf cubes).
        std::vector<TGVCube> state_leaf;
        {
            std::ifstream sin(tgv_state_path, std::ios::binary);
            if (!sin) throw std::runtime_error("Cannot open: " + tgv_state_path);
            sin.seekg(12 + static_cast<std::streamoff>(leaf.first_idx) * sizeof(TGVCube));
            state_leaf.resize(cubes_leaf.size());
            sin.read(reinterpret_cast<char*>(state_leaf.data()),
                     state_leaf.size() * sizeof(TGVCube));
        }

        // CTF lookup helpers, scoped to one open file per leaf.
        CubeReader     cr(balanced_cube_path);
        TgvStateReader sr(tgv_state_path);
        HistReader     hr(hist_path);

        // Small in-leaf code → local index map (covers all depths in the leaf).
        std::unordered_map<uint64_t, uint32_t> leaf_map;
        leaf_map.reserve(cubes_leaf.size());
        for (uint32_t li = 0; li < cubes_leaf.size(); ++li)
            leaf_map[mc_key2(cubes_leaf[li].code)] = li;

        // Cache for repeated cross-leaf border lookups within this leaf.
        struct BorderInfo {
            uint32_t   gidx;
            OctreeCube cube;
            float      u;
            uint32_t   hist_total;
        };
        std::unordered_map<uint64_t, BorderInfo> border_cache;

        std::vector<QuadRec> quads;
        quads.reserve(cubes_leaf.size() * 2);

        auto resolve_border = [&](const MortonCode& target) -> const BorderInfo* {
            uint64_t k = mc_key2(target);
            auto it = border_cache.find(k);
            if (it != border_cache.end())
                return it->second.gidx == UINT32_MAX ? nullptr : &it->second;
            uint32_t g = cr.lookup(target);
            BorderInfo bi;
            bi.gidx = g;
            if (g != UINT32_MAX) {
                bi.cube       = cr.read_at(g);
                bi.u          = sr.read_at(g).u;
                bi.hist_total = hr.read_at(g).total();
            }
            border_cache.emplace(k, bi);
            return g == UINT32_MAX ? nullptr : &border_cache.find(k)->second;
        };

        for (uint32_t li = 0; li < cubes_leaf.size(); ++li) {
            const OctreeCube& ci = cubes_leaf[li];
            const uint32_t    gi = leaf.first_idx + li;
            const float       ui = state_leaf[li].u;
            const uint32_t    hi_total = hists_leaf[li].total();

            for (int f = 0; f < 6; ++f) {
                MortonCode nb = face_neighbor(ci.code, f, ci.depth);
                if (nb.hi == UINT32_MAX) continue;

                // Resolve neighbour by trying, in order: in-leaf exact, cross-leaf
                // exact, in-leaf parent, cross-leaf parent.  parent_code(nb, depth)
                // for a same-depth neighbour can numerically equal ci's own code
                // when nb's lower Morton bits are below the parent-clear mask
                // (e.g. ci at origin, nb a +x cell at the same depth); in that
                // case the parent fallback is a spurious self-match, so skip
                // any candidate equal to gi and try the next option.
                OctreeCube cj{}; uint32_t gj = UINT32_MAX;
                float uj = 0.0f; uint32_t hj_total = 0;

                auto take_in_leaf = [&](const MortonCode& code) {
                    if (gj != UINT32_MAX) return;
                    auto it = leaf_map.find(mc_key2(code));
                    if (it == leaf_map.end()) return;
                    uint32_t lj = it->second;
                    uint32_t cand = leaf.first_idx + lj;
                    if (cand == gi) return;
                    cj = cubes_leaf[lj]; gj = cand;
                    uj = state_leaf[lj].u; hj_total = hists_leaf[lj].total();
                };
                auto take_cross_leaf = [&](const MortonCode& code) {
                    if (gj != UINT32_MAX) return;
                    const BorderInfo* bi = resolve_border(code);
                    if (!bi) return;
                    if (bi->gidx == gi) return;
                    cj = bi->cube; gj = bi->gidx;
                    uj = bi->u; hj_total = bi->hist_total;
                };

                take_in_leaf(nb);
                take_cross_leaf(nb);
                if (ci.depth > 0) {
                    MortonCode par = parent_code(nb, ci.depth);
                    take_in_leaf(par);
                    take_cross_leaf(par);
                }
                if (gj == UINT32_MAX) continue;

                // Ownership: finer side wins (CTF only fires from finer side);
                // equal depth → smaller global index wins.
                if (cj.depth > ci.depth) continue;
                if (cj.depth == ci.depth && gi > gj) continue;

                if (hi_total == 0 || hj_total == 0) continue;
                if ((ui >= 0.0f) == (uj >= 0.0f)) continue;

                float cs_i = 2.0f * r_root / static_cast<float>(1u << ci.depth);
                float cs_j = 2.0f * r_root / static_cast<float>(1u << cj.depth);
                float cs_min = (cs_i < cs_j) ? cs_i : cs_j;
                float h      = cs_min * 0.5f;

                Vec3f pi = cube_corner2(ci, origin, r_root);
                Vec3f pj = cube_corner2(cj, origin, r_root);

                float t = ui / (ui - uj);
                int norm_ax = f / 2;
                Vec3f surf;
                surf[norm_ax] = (pi[norm_ax] + cs_i * 0.5f)
                              + t * ((pj[norm_ax] + cs_j * 0.5f)
                                   - (pi[norm_ax] + cs_i * 0.5f));
                const Vec3f& small_corner = (cs_i <= cs_j) ? pi : pj;
                switch (norm_ax) {
                    case 0: surf[1] = small_corner[1] + h; surf[2] = small_corner[2] + h; break;
                    case 1: surf[0] = small_corner[0] + h; surf[2] = small_corner[2] + h; break;
                    case 2: surf[0] = small_corner[0] + h; surf[1] = small_corner[1] + h; break;
                }
                Vec3f a1 = Vec3f::Zero(), a2 = Vec3f::Zero();
                switch (norm_ax) {
                    case 0: a1[1] = h; a2[2] = h; break;
                    case 1: a1[2] = h; a2[0] = h; break;
                    case 2: a1[0] = h; a2[1] = h; break;
                }
                Vec3f v0 = surf - a1 - a2;
                Vec3f v1 = surf + a1 - a2;
                Vec3f v2 = surf + a1 + a2;
                Vec3f v3 = surf - a1 + a2;

                bool face_pos = (f % 2 == 0);
                bool flip     = face_pos ? (ui > 0.0f) : (ui < 0.0f);

                QuadRec q{};
                q.key[0] = vkey(v0, origin, inv_scale);
                q.key[1] = vkey(v1, origin, inv_scale);
                q.key[2] = vkey(v2, origin, inv_scale);
                q.key[3] = vkey(v3, origin, inv_scale);
                q.flip   = flip ? 1u : 0u;
                quads.push_back(q);
            }
        }

        if (!quads.empty())
            qout.write(reinterpret_cast<const char*>(quads.data()),
                       quads.size() * sizeof(QuadRec));
    }
    qout.close();

    // -------------------------------------------------------------------------
    // Pass B: aggregate. Scan quads file once → collect unique vertex keys,
    // assign indices. Open OBJ. Write all v lines. Re-scan quads → write f lines.
    // -------------------------------------------------------------------------
    std::ifstream qin(quads_path, std::ios::binary);
    if (!qin) throw std::runtime_error("Cannot open: " + quads_path);

    std::unordered_map<uint64_t, uint32_t> vmap;

    {
        constexpr size_t CHUNK = 1u << 14;
        std::vector<QuadRec> buf(CHUNK);
        while (qin) {
            qin.read(reinterpret_cast<char*>(buf.data()), CHUNK * sizeof(QuadRec));
            size_t got = qin.gcount() / sizeof(QuadRec);
            for (size_t i = 0; i < got; ++i)
                for (int k = 0; k < 4; ++k) {
                    auto [it, inserted] = vmap.try_emplace(buf[i].key[k],
                                                            (uint32_t)vmap.size());
                    (void)it; (void)inserted;
                }
            if (got < CHUNK) break;
        }
    }

    std::ofstream obj(out_obj_path, std::ios::trunc);
    if (!obj) throw std::runtime_error("Cannot open: " + out_obj_path);

    // Write v lines in ascending-index order.
    std::vector<uint64_t> keys_by_idx(vmap.size());
    for (auto& [k, idx] : vmap) keys_by_idx[idx] = k;
    for (size_t i = 0; i < keys_by_idx.size(); ++i) {
        Vec3f v = dequantise(keys_by_idx[i], origin, scale);
        obj << "v " << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';
    }

    // Re-scan quads → triangle faces.
    qin.clear();
    qin.seekg(0);
    {
        constexpr size_t CHUNK = 1u << 14;
        std::vector<QuadRec> buf(CHUNK);
        while (qin) {
            qin.read(reinterpret_cast<char*>(buf.data()), CHUNK * sizeof(QuadRec));
            size_t got = qin.gcount() / sizeof(QuadRec);
            for (size_t i = 0; i < got; ++i) {
                uint32_t i0 = vmap[buf[i].key[0]];
                uint32_t i1 = vmap[buf[i].key[1]];
                uint32_t i2 = vmap[buf[i].key[2]];
                uint32_t i3 = vmap[buf[i].key[3]];
                if (!buf[i].flip) {
                    obj << "f " << (i0+1) << ' ' << (i1+1) << ' ' << (i2+1) << '\n';
                    obj << "f " << (i0+1) << ' ' << (i2+1) << ' ' << (i3+1) << '\n';
                } else {
                    obj << "f " << (i0+1) << ' ' << (i2+1) << ' ' << (i1+1) << '\n';
                    obj << "f " << (i0+1) << ' ' << (i3+1) << ' ' << (i2+1) << '\n';
                }
            }
            if (got < CHUNK) break;
        }
    }
    qin.close();
    std::remove(quads_path.c_str());
}

// ---------- OBJ output -------------------------------------------------------

void write_obj(const std::string& path, const Mesh& mesh) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open: " + path);

    for (const auto& v : mesh.verts)
        out << "v " << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';

    if (!mesh.normals.empty())
        for (const auto& n : mesh.normals)
            out << "vn " << n.x() << ' ' << n.y() << ' ' << n.z() << '\n';

    const size_t nt = mesh.tri_count();
    if (mesh.normals.empty()) {
        for (size_t t = 0; t < nt; ++t) {
            uint32_t a = mesh.faces[t*3] + 1;
            uint32_t b = mesh.faces[t*3+1] + 1;
            uint32_t c = mesh.faces[t*3+2] + 1;
            out << "f " << a << ' ' << b << ' ' << c << '\n';
        }
    } else {
        for (size_t t = 0; t < nt; ++t) {
            uint32_t a = mesh.faces[t*3] + 1;
            uint32_t b = mesh.faces[t*3+1] + 1;
            uint32_t c = mesh.faces[t*3+2] + 1;
            out << "f " << a << "//" << a << ' '
                        << b << "//" << b << ' '
                        << c << "//" << c << '\n';
        }
    }
}
