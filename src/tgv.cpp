#include "tgv.hpp"
#include "balance.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// ---------- file I/O ---------------------------------------------------------

namespace {
struct TGVFileHeader {
    char     magic[4] = {'T', 'G', 'V', 'S'};
    uint32_t version  = 1;
    uint32_t count    = 0;
};
}

void save_tgv(const std::string& path, const std::vector<TGVCube>& state) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open: " + path);
    TGVFileHeader hdr;
    hdr.count = static_cast<uint32_t>(state.size());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!state.empty())
        out.write(reinterpret_cast<const char*>(state.data()),
                  state.size() * sizeof(TGVCube));
}

std::vector<TGVCube> load_tgv(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    TGVFileHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='S')
        throw std::runtime_error("Bad TGV state magic");
    std::vector<TGVCube> s(hdr.count);
    if (hdr.count > 0)
        in.read(reinterpret_cast<char*>(s.data()), hdr.count * sizeof(TGVCube));
    return s;
}

// ---------- helpers ----------------------------------------------------------

using CodeMap = std::unordered_map<uint64_t, uint32_t>;

static uint64_t key(const MortonCode& m) {
    return (static_cast<uint64_t>(m.hi) << 32) | m.lo;
}

// Project L2-ball of radius r.
static void clip_ball3(float p[3], float r) {
    float n2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
    if (n2 > r * r) {
        float s = r / std::sqrt(n2);
        p[0] *= s; p[1] *= s; p[2] *= s;
    }
}

static void clip_ball6(float q[6], float r) {
    float n2 = 0;
    for (int i = 0; i < 6; ++i) n2 += q[i]*q[i];
    if (n2 > r * r) {
        float s = r / std::sqrt(n2);
        for (int i = 0; i < 6; ++i) q[i] *= s;
    }
}

// Proximal of τ*Σ_k w_k |u - a_k| (weighted sum of L1).
static float prox_data(float z, float tau, const CubeHistogram& hist) {
    uint32_t tot = hist.total();
    if (tot == 0) return z;

    uint32_t cum = 0;
    float med = z;
    for (int k = 0; k < 8; ++k) {
        cum += hist.bins[k];
        if (cum * 2 >= tot) {
            med = -1.0f + (k + 0.5f) / 4.0f;
            break;
        }
    }

    float diff = z - med;
    float shrink = tau * static_cast<float>(tot);
    if      (diff >  shrink) return z - shrink;
    else if (diff < -shrink) return z + shrink;
    else                     return med;
}

// Neighbour lookup with coarser fallback (coarse-to-fine).
static int32_t neighbor_idx_ctf(const CodeMap& cmap,
                                 const MortonCode& code, int face, int depth) {
    MortonCode nb = face_neighbor(code, face, depth);
    if (nb.hi == UINT32_MAX) return -1;
    auto it = cmap.find(key(nb));
    if (it != cmap.end()) return static_cast<int32_t>(it->second);
    if (depth > 0) {
        MortonCode nb_par = parent_code(nb, depth);
        it = cmap.find(key(nb_par));
        if (it != cmap.end()) return static_cast<int32_t>(it->second);
    }
    return -1;
}

// ---------- precomputed neighbour table --------------------------------------
// For each active cube, store its 6 neighbour indices (or -1) once before
// the iteration loop, eliminating repeated hash-map lookups in the hot path.

struct NeighbourTable {
    // nb[i*6 + f] = neighbour index of active[i] in face direction f, or -1.
    std::vector<int32_t> nb;

    void build(const std::vector<uint32_t>& active,
               const std::vector<OctreeCube>& cubes,
               const CodeMap& cmap) {
        const size_t n = active.size();
        nb.resize(n * 6);
        for (size_t ai = 0; ai < n; ++ai) {
            uint32_t i = active[ai];
            const OctreeCube& c = cubes[i];
            for (int f = 0; f < 6; ++f)
                nb[ai * 6 + f] = neighbor_idx_ctf(cmap, c.code, f, c.depth);
        }
    }

    int32_t get(size_t ai, int f) const { return nb[ai * 6 + f]; }
};

// ---------- one level of primal-dual iterations ------------------------------

static void run_level(
    const std::vector<uint32_t>& active,
    const std::vector<OctreeCube>& /*cubes*/,
    const std::vector<CubeHistogram>& hists,
    std::vector<TGVCube>& state,
    std::vector<float>& u_bar,
    std::vector<float>& vx_bar,
    std::vector<float>& vy_bar,
    std::vector<float>& vz_bar,
    const NeighbourTable& nbt,
    const TGVParams& params)
{
    const float sigma = 0.25f;
    const float tau   = 0.25f;
    const size_t na   = active.size();

    std::vector<float> u_new(na);
    std::vector<float> vx_new(na), vy_new(na), vz_new(na);

    for (int iter = 0; iter < params.iters; ++iter) {

        // ---- dual p ---------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active[ai];
            auto ub = [&](int f) -> float {
                int32_t nb = nbt.get(ai, f);
                return nb >= 0 ? u_bar[nb] : u_bar[i];
            };
            state[i].p[0] += sigma * (ub(0) - u_bar[i] - vx_bar[i]);
            state[i].p[1] += sigma * (ub(2) - u_bar[i] - vy_bar[i]);
            state[i].p[2] += sigma * (ub(4) - u_bar[i] - vz_bar[i]);
            clip_ball3(state[i].p, params.alpha1);
        }

        // ---- dual q ---------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active[ai];
            auto vb = [&](int f, int comp) -> float {
                int32_t nb = nbt.get(ai, f);
                if (nb < 0) return comp==0 ? vx_bar[i] : comp==1 ? vy_bar[i] : vz_bar[i];
                return comp==0 ? vx_bar[nb] : comp==1 ? vy_bar[nb] : vz_bar[nb];
            };
            float eps[6] = {
                vb(0,0) - vx_bar[i],
                vb(2,1) - vy_bar[i],
                vb(4,2) - vz_bar[i],
                0.5f*(vb(2,0)-vx_bar[i] + vb(0,1)-vy_bar[i]),
                0.5f*(vb(4,0)-vx_bar[i] + vb(0,2)-vz_bar[i]),
                0.5f*(vb(4,1)-vy_bar[i] + vb(2,2)-vz_bar[i])
            };
            for (int k = 0; k < 6; ++k) state[i].q[k] += sigma * eps[k];
            clip_ball6(state[i].q, params.alpha0);
        }

        // ---- primal u -------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active[ai];
            auto gp = [&](int f, int comp) -> float {
                int32_t nb = nbt.get(ai, f);
                return nb >= 0 ? state[nb].p[comp] : 0.0f;
            };
            float div_p = (state[i].p[0] - gp(1,0))
                        + (state[i].p[1] - gp(3,1))
                        + (state[i].p[2] - gp(5,2));
            u_new[ai] = prox_data(state[i].u + tau * div_p, tau, hists[i]);
        }

        // ---- primal v -------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active[ai];
            auto gq = [&](int f, int comp) -> float {
                int32_t nb = nbt.get(ai, f);
                return nb >= 0 ? state[nb].q[comp] : 0.0f;
            };
            float dqx = (state[i].q[0]-gq(1,0)) + (state[i].q[3]-gq(3,3)) + (state[i].q[4]-gq(5,4));
            float dqy = (state[i].q[3]-gq(1,3)) + (state[i].q[1]-gq(3,1)) + (state[i].q[5]-gq(5,5));
            float dqz = (state[i].q[4]-gq(1,4)) + (state[i].q[5]-gq(3,5)) + (state[i].q[2]-gq(5,2));
            vx_new[ai] = state[i].v[0] + tau * (state[i].p[0] + dqx);
            vy_new[ai] = state[i].v[1] + tau * (state[i].p[1] + dqy);
            vz_new[ai] = state[i].v[2] + tau * (state[i].p[2] + dqz);
        }

        // ---- extrapolation --------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active[ai];
            u_bar[i]  = 2.0f*u_new[ai]  - state[i].u;
            vx_bar[i] = 2.0f*vx_new[ai] - state[i].v[0];
            vy_bar[i] = 2.0f*vy_new[ai] - state[i].v[1];
            vz_bar[i] = 2.0f*vz_new[ai] - state[i].v[2];
            state[i].u    = u_new[ai];
            state[i].v[0] = vx_new[ai];
            state[i].v[1] = vy_new[ai];
            state[i].v[2] = vz_new[ai];
        }
    }
}

// ---------- main TGV minimization (coarse-to-fine) ---------------------------

std::vector<TGVCube> tgv_minimize(
    const std::vector<OctreeCube>& cubes,
    const std::vector<CubeHistogram>& hists,
    const Vec3f& /*origin*/,
    float r_root,
    const TGVParams& params)
{
    const uint32_t N = static_cast<uint32_t>(cubes.size());
    std::vector<TGVCube> state(N);
    (void)r_root;

    CodeMap cmap;
    cmap.reserve(N);
    for (uint32_t i = 0; i < N; ++i)
        cmap[key(cubes[i].code)] = i;

    for (uint32_t i = 0; i < N; ++i)
        state[i].u = hists[i].median_indicator();

    int min_depth = 21, max_depth = 0;
    for (auto& c : cubes) {
        min_depth = std::min(min_depth, c.depth);
        max_depth = std::max(max_depth, c.depth);
    }
    std::vector<std::vector<uint32_t>> by_depth(max_depth + 1);
    for (uint32_t i = 0; i < N; ++i)
        by_depth[cubes[i].depth].push_back(i);

    std::vector<float> u_bar(N), vx_bar(N), vy_bar(N), vz_bar(N);
    for (uint32_t i = 0; i < N; ++i) u_bar[i] = state[i].u;

    for (int L = min_depth; L <= max_depth; ++L) {
        const auto& active = by_depth[L];
        if (active.empty()) continue;

        if (L > min_depth) {
            for (uint32_t i : active) {
                MortonCode par = parent_code(cubes[i].code, L);
                auto it = cmap.find(key(par));
                if (it != cmap.end()) {
                    uint32_t pi = it->second;
                    state[i].u    = state[pi].u;
                    state[i].v[0] = state[pi].v[0];
                    state[i].v[1] = state[pi].v[1];
                    state[i].v[2] = state[pi].v[2];
                }
                state[i].p[0] = state[i].p[1] = state[i].p[2] = 0.0f;
                for (int k = 0; k < 6; ++k) state[i].q[k] = 0.0f;
                u_bar[i]  = state[i].u;
                vx_bar[i] = state[i].v[0];
                vy_bar[i] = state[i].v[1];
                vz_bar[i] = state[i].v[2];
            }
        }

        // Build neighbour table once per level — amortises hash-map cost over all iters.
        NeighbourTable nbt;
        nbt.build(active, cubes, cmap);

        run_level(active, cubes, hists, state, u_bar, vx_bar, vy_bar, vz_bar, nbt, params);
    }

    return state;
}
