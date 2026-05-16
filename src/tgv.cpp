#include "tgv.hpp"
#include "balance.hpp"
#include "octree_builder.hpp"
#include "treetop.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <cstring>

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

// ---------- one level of primal-dual iterations (frozen-border variant) ------
// Identical to run_level except that out-of-leaf neighbor lookups for u_bar
// and v_bar use pre-snapshot frozen values rather than the live arrays.
// "In-leaf" is defined as: leaf_first_idx <= nb <= leaf_last_idx.
// p/q of out-of-leaf cubes are read as-is (they are 0 at level start and
// accumulate only within their own leaf, so cross-leaf contamination is minor).

static void run_level_frozen(
    const std::vector<uint32_t>& active,
    const std::vector<OctreeCube>& /*cubes*/,
    const std::vector<CubeHistogram>& hists,
    std::vector<TGVCube>& state,
    std::vector<float>& u_bar,
    std::vector<float>& vx_bar,
    std::vector<float>& vy_bar,
    std::vector<float>& vz_bar,
    const std::vector<float>& frozen_u,
    const std::vector<float>& frozen_vx,
    const std::vector<float>& frozen_vy,
    const std::vector<float>& frozen_vz,
    uint32_t leaf_first_idx,
    uint32_t leaf_last_idx,
    const NeighbourTable& nbt,
    const TGVParams& params)
{
    const float sigma = 0.25f;
    const float tau   = 0.25f;
    const size_t na   = active.size();

    auto in_leaf = [&](int32_t nb) -> bool {
        if (nb < 0) return false;
        auto u = static_cast<uint32_t>(nb);
        return u >= leaf_first_idx && u <= leaf_last_idx;
    };

    std::vector<float> u_new(na);
    std::vector<float> vx_new(na), vy_new(na), vz_new(na);

    for (int iter = 0; iter < params.iters; ++iter) {

        // ---- dual p ---------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active[ai];
            auto ub = [&](int f) -> float {
                int32_t nb = nbt.get(ai, f);
                if (nb < 0) return u_bar[i];
                return in_leaf(nb) ? u_bar[nb] : frozen_u[nb];
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
                if (nb < 0)
                    return comp==0 ? vx_bar[i] : comp==1 ? vy_bar[i] : vz_bar[i];
                if (!in_leaf(nb))
                    return comp==0 ? frozen_vx[nb] : comp==1 ? frozen_vy[nb] : frozen_vz[nb];
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

// ---------- out-of-core TGV minimization (per treetop leaf) ------------------

std::vector<TGVCube> tgv_minimize_oc(
    const std::vector<OctreeCube>& cubes,
    const std::vector<CubeHistogram>& hists,
    const std::vector<TreetopLeaf>& leaves,
    const Vec3f& /*origin*/,
    float r_root,
    const TGVParams& params)
{
    const uint32_t N = static_cast<uint32_t>(cubes.size());
    std::vector<TGVCube> state(N);
    (void)r_root;

    if (N == 0) return state;

    CodeMap cmap;
    cmap.reserve(N);
    for (uint32_t i = 0; i < N; ++i)
        cmap[key(cubes[i].code)] = i;

    for (uint32_t i = 0; i < N; ++i)
        state[i].u = hists[i].median_indicator();

    int min_depth = 21, max_depth = 0;
    for (const auto& c : cubes) {
        min_depth = std::min(min_depth, c.depth);
        max_depth = std::max(max_depth, c.depth);
    }
    std::vector<std::vector<uint32_t>> by_depth(max_depth + 1);
    for (uint32_t i = 0; i < N; ++i)
        by_depth[cubes[i].depth].push_back(i);

    std::vector<float> u_bar(N), vx_bar(N), vy_bar(N), vz_bar(N);
    for (uint32_t i = 0; i < N; ++i) u_bar[i] = state[i].u;

    for (int L = min_depth; L <= max_depth; ++L) {
        const auto& level_all = by_depth[L];
        if (level_all.empty()) continue;

        // Initialize level-L cubes from parent (coarse-to-fine).
        if (L > min_depth) {
            for (uint32_t i : level_all) {
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

        // Snapshot primal bars before any leaf at this level runs.
        // Out-of-leaf lookups use these frozen values so each leaf is independent.
        const std::vector<float> frozen_u  = u_bar;
        const std::vector<float> frozen_vx = vx_bar;
        const std::vector<float> frozen_vy = vy_bar;
        const std::vector<float> frozen_vz = vz_bar;

        for (const auto& leaf : leaves) {
            // Extract the subset of level_all that falls within this leaf's range.
            // level_all is in ascending index order (populated via a 0..N-1 loop),
            // so binary search is valid.
            auto lo = std::lower_bound(level_all.begin(), level_all.end(), leaf.first_idx);
            auto hi = std::upper_bound(level_all.begin(), level_all.end(), leaf.last_idx);
            if (lo == hi) continue;

            std::vector<uint32_t> leaf_active(lo, hi);

            NeighbourTable nbt;
            nbt.build(leaf_active, cubes, cmap);

            run_level_frozen(leaf_active, cubes, hists, state,
                             u_bar, vx_bar, vy_bar, vz_bar,
                             frozen_u, frozen_vx, frozen_vy, frozen_vz,
                             leaf.first_idx, leaf.last_idx,
                             nbt, params);
        }
    }

    return state;
}

// ============================================================================
// Streaming out-of-core TGV minimization
// ============================================================================
//
// All large arrays live on disk; only data for the currently-processed leaf
// (plus its 1-ring border) is in RAM.
//
//   tgv_state.bin   – TGVCube[N]      (current primal+dual state)
//   bars_frozen.bin – BarsFrozen[N]   (u/vx/vy/vz snapshot at level start)
//
// Each depth level snapshots state→bars_frozen, then iterates leaves
// independently.  Border u_bar/v_bar values come from the frozen file;
// border p/q are treated as zero (cross-leaf coupling is intentionally
// suppressed to guarantee seam-free reconstruction).

namespace {

struct BarsFrozen {
    float u  = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
};

// ---------- low-level TGV file I/O ------------------------------------------

void init_tgv_file(const std::string& path, uint32_t N) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open: " + path);
    TGVFileHeader hdr;
    hdr.count = N;
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (N == 0) return;
    // Extend the file by writing N zero-initialised TGVCubes in chunks.
    constexpr size_t CHUNK = 1u << 16;
    std::vector<TGVCube> zeros(std::min<size_t>(N, CHUNK));
    size_t written = 0;
    while (written < N) {
        size_t k = std::min<size_t>(CHUNK, N - written);
        out.write(reinterpret_cast<const char*>(zeros.data()), k * sizeof(TGVCube));
        written += k;
    }
}

std::vector<TGVCube> load_tgv_range(const std::string& path,
                                    uint32_t first_idx, uint32_t last_idx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    TGVFileHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (first_idx >= hdr.count || first_idx > last_idx) return {};
    uint32_t end = std::min(last_idx + 1, hdr.count);
    uint32_t count = end - first_idx;
    in.seekg(sizeof(TGVFileHeader) +
             static_cast<std::streamoff>(first_idx) * sizeof(TGVCube));
    std::vector<TGVCube> s(count);
    in.read(reinterpret_cast<char*>(s.data()), count * sizeof(TGVCube));
    return s;
}

void save_tgv_range(const std::string& path, uint32_t first_idx,
                    const std::vector<TGVCube>& s) {
    if (s.empty()) return;
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!io) throw std::runtime_error("Cannot open: " + path);
    io.seekp(sizeof(TGVFileHeader) +
             static_cast<std::streamoff>(first_idx) * sizeof(TGVCube));
    io.write(reinterpret_cast<const char*>(s.data()), s.size() * sizeof(TGVCube));
}

// Random-access read of arbitrary indices (used for border cubes / parents).
std::vector<TGVCube> load_tgv_at(const std::string& path,
                                 const std::vector<uint32_t>& indices) {
    std::vector<TGVCube> out(indices.size());
    if (indices.empty()) return out;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    for (size_t i = 0; i < indices.size(); ++i) {
        in.seekg(sizeof(TGVFileHeader) +
                 static_cast<std::streamoff>(indices[i]) * sizeof(TGVCube));
        in.read(reinterpret_cast<char*>(&out[i]), sizeof(TGVCube));
    }
    return out;
}

// ---------- frozen-bars file ------------------------------------------------

struct BarsFileHeader {
    char     magic[4] = {'T', 'G', 'V', 'B'};
    uint32_t version  = 1;
    uint32_t count    = 0;
};

void build_bars_frozen(const std::string& state_path,
                       const std::string& bars_path) {
    std::ifstream in(state_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + state_path);
    TGVFileHeader sh;
    in.read(reinterpret_cast<char*>(&sh), sizeof(sh));

    std::ofstream out(bars_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open: " + bars_path);
    BarsFileHeader bh;
    bh.count = sh.count;
    out.write(reinterpret_cast<const char*>(&bh), sizeof(bh));

    constexpr size_t CHUNK = 1u << 16;
    std::vector<TGVCube> sbuf(CHUNK);
    std::vector<BarsFrozen> bbuf(CHUNK);
    uint32_t remaining = sh.count;
    while (remaining > 0) {
        size_t k = std::min<size_t>(CHUNK, remaining);
        in.read(reinterpret_cast<char*>(sbuf.data()), k * sizeof(TGVCube));
        for (size_t i = 0; i < k; ++i) {
            bbuf[i].u  = sbuf[i].u;
            bbuf[i].vx = sbuf[i].v[0];
            bbuf[i].vy = sbuf[i].v[1];
            bbuf[i].vz = sbuf[i].v[2];
        }
        out.write(reinterpret_cast<const char*>(bbuf.data()), k * sizeof(BarsFrozen));
        remaining -= static_cast<uint32_t>(k);
    }
}

std::vector<BarsFrozen> load_bars_at(const std::string& path,
                                     const std::vector<uint32_t>& indices) {
    std::vector<BarsFrozen> out(indices.size());
    if (indices.empty()) return out;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    for (size_t i = 0; i < indices.size(); ++i) {
        in.seekg(sizeof(BarsFileHeader) +
                 static_cast<std::streamoff>(indices[i]) * sizeof(BarsFrozen));
        in.read(reinterpret_cast<char*>(&out[i]), sizeof(BarsFrozen));
    }
    return out;
}

// ---------- Morton-code lookup in the sorted cube file ----------------------

class CubeFileReader {
public:
    explicit CubeFileReader(const std::string& path) : in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("Cannot open: " + path);
        in_.read(reinterpret_cast<char*>(&hdr_), sizeof(hdr_));
        if (hdr_.magic[0]!='T'||hdr_.magic[1]!='G'||hdr_.magic[2]!='V'||hdr_.magic[3]!='1')
            throw std::runtime_error("Bad cube file magic: " + path);
    }
    uint32_t count() const { return hdr_.cube_count; }
    float    r_root() const { return hdr_.r_root; }

    OctreeCube read_at(uint32_t idx) {
        in_.seekg(sizeof(CubeFileHeader) +
                  static_cast<std::streamoff>(idx) * sizeof(OctreeCube));
        OctreeCube c;
        in_.read(reinterpret_cast<char*>(&c), sizeof(c));
        return c;
    }

    // Returns index whose code == target, or UINT32_MAX if not present.
    uint32_t lookup(const MortonCode& target) {
        uint32_t lo = 0, hi = hdr_.cube_count;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            OctreeCube c = read_at(mid);
            if (c.code == target) return mid;
            if (c.code < target) lo = mid + 1;
            else                 hi = mid;
        }
        return UINT32_MAX;
    }

private:
    std::ifstream    in_;
    CubeFileHeader   hdr_;
};

// ---------- streaming primal-dual core ---------------------------------------
//
// `state_local[0 .. n_leaf-1]`        — every cube in the current leaf
// `state_local[n_leaf .. n_leaf+n_border-1]` — 1-ring border cubes outside the leaf
// `active_pos` indexes the subset of [0, n_leaf) that's at the current depth L —
//   only these get updated.  Out-of-active reads use whatever value is in the
//   arrays (in-leaf coarser cubes carry their result from prior levels;
//   border cubes carry their frozen bars and zero p/q).

struct LocalNbt {
    std::vector<int32_t> nb;            // size = active_pos.size() * 6
    int32_t get(size_t ai, int f) const { return nb[ai * 6 + f]; }
};

void run_level_streaming(
    const std::vector<uint32_t>& active_pos,
    const std::vector<CubeHistogram>& hists_leaf,   // size = n_leaf
    std::vector<TGVCube>& state_local,
    std::vector<float>& u_bar,
    std::vector<float>& vx_bar,
    std::vector<float>& vy_bar,
    std::vector<float>& vz_bar,
    const LocalNbt& nbt,
    const TGVParams& params)
{
    const float sigma = 0.25f;
    const float tau   = 0.25f;
    const size_t na   = active_pos.size();

    std::vector<float> u_new(na);
    std::vector<float> vx_new(na), vy_new(na), vz_new(na);

    for (int iter = 0; iter < params.iters; ++iter) {

        // ---- dual p ---------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active_pos[ai];
            auto ub = [&](int f) -> float {
                int32_t nb = nbt.get(ai, f);
                return nb >= 0 ? u_bar[nb] : u_bar[i];
            };
            state_local[i].p[0] += sigma * (ub(0) - u_bar[i] - vx_bar[i]);
            state_local[i].p[1] += sigma * (ub(2) - u_bar[i] - vy_bar[i]);
            state_local[i].p[2] += sigma * (ub(4) - u_bar[i] - vz_bar[i]);
            clip_ball3(state_local[i].p, params.alpha1);
        }

        // ---- dual q ---------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active_pos[ai];
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
            for (int k = 0; k < 6; ++k) state_local[i].q[k] += sigma * eps[k];
            clip_ball6(state_local[i].q, params.alpha0);
        }

        // ---- primal u -------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active_pos[ai];
            auto gp = [&](int f, int comp) -> float {
                int32_t nb = nbt.get(ai, f);
                return nb >= 0 ? state_local[nb].p[comp] : 0.0f;
            };
            float div_p = (state_local[i].p[0] - gp(1,0))
                        + (state_local[i].p[1] - gp(3,1))
                        + (state_local[i].p[2] - gp(5,2));
            u_new[ai] = prox_data(state_local[i].u + tau * div_p, tau, hists_leaf[i]);
        }

        // ---- primal v -------------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active_pos[ai];
            auto gq = [&](int f, int comp) -> float {
                int32_t nb = nbt.get(ai, f);
                return nb >= 0 ? state_local[nb].q[comp] : 0.0f;
            };
            float dqx = (state_local[i].q[0]-gq(1,0)) + (state_local[i].q[3]-gq(3,3)) + (state_local[i].q[4]-gq(5,4));
            float dqy = (state_local[i].q[3]-gq(1,3)) + (state_local[i].q[1]-gq(3,1)) + (state_local[i].q[5]-gq(5,5));
            float dqz = (state_local[i].q[4]-gq(1,4)) + (state_local[i].q[5]-gq(3,5)) + (state_local[i].q[2]-gq(5,2));
            vx_new[ai] = state_local[i].v[0] + tau * (state_local[i].p[0] + dqx);
            vy_new[ai] = state_local[i].v[1] + tau * (state_local[i].p[1] + dqy);
            vz_new[ai] = state_local[i].v[2] + tau * (state_local[i].p[2] + dqz);
        }

        // ---- extrapolation --------------------------------------------------
        for (size_t ai = 0; ai < na; ++ai) {
            uint32_t i = active_pos[ai];
            u_bar[i]  = 2.0f*u_new[ai]  - state_local[i].u;
            vx_bar[i] = 2.0f*vx_new[ai] - state_local[i].v[0];
            vy_bar[i] = 2.0f*vy_new[ai] - state_local[i].v[1];
            vz_bar[i] = 2.0f*vz_new[ai] - state_local[i].v[2];
            state_local[i].u    = u_new[ai];
            state_local[i].v[0] = vx_new[ai];
            state_local[i].v[1] = vy_new[ai];
            state_local[i].v[2] = vz_new[ai];
        }
    }
}

} // namespace

void tgv_minimize_streaming(
    const std::string& balanced_cube_path,
    const std::string& hist_path,
    const std::string& treetop_path,
    const Vec3f&       /*origin*/,
    float              r_root,
    const std::string& out_state_path,
    const TGVParams&   params)
{
    (void)r_root;
    float r_root_tt = 0.0f;
    auto leaves = load_treetop(treetop_path, r_root_tt);

    // ---- determine total cube count --------------------------------------
    uint32_t total_cubes = 0;
    {
        CubeFileReader cr(balanced_cube_path);
        total_cubes = cr.count();
    }
    init_tgv_file(out_state_path, total_cubes);
    if (total_cubes == 0) return;

    // ---- seed state.u = median_indicator(hists), streamed -----------------
    {
        std::ifstream hin(hist_path, std::ios::binary);
        if (!hin) throw std::runtime_error("Cannot open: " + hist_path);
        // Skip histogram file header (TGVH/version/count): 12 bytes.
        hin.seekg(12);
        std::fstream sio(out_state_path, std::ios::binary | std::ios::in | std::ios::out);
        sio.seekp(sizeof(TGVFileHeader));

        constexpr size_t CHUNK = 1u << 16;
        std::vector<CubeHistogram> hbuf(CHUNK);
        std::vector<TGVCube>       sbuf(CHUNK);
        uint32_t remaining = total_cubes;
        while (remaining > 0) {
            size_t k = std::min<size_t>(CHUNK, remaining);
            hin.read(reinterpret_cast<char*>(hbuf.data()), k * sizeof(CubeHistogram));
            for (size_t i = 0; i < k; ++i) {
                sbuf[i] = TGVCube{};
                sbuf[i].u = hbuf[i].median_indicator();
            }
            sio.write(reinterpret_cast<const char*>(sbuf.data()), k * sizeof(TGVCube));
            remaining -= static_cast<uint32_t>(k);
        }
    }

    // ---- discover depth range by scanning the cube file ------------------
    int min_depth = 21, max_depth = 0;
    {
        std::ifstream in(balanced_cube_path, std::ios::binary);
        in.seekg(sizeof(CubeFileHeader));
        constexpr size_t CHUNK = 1u << 16;
        std::vector<OctreeCube> buf(CHUNK);
        uint32_t remaining = total_cubes;
        while (remaining > 0) {
            size_t k = std::min<size_t>(CHUNK, remaining);
            in.read(reinterpret_cast<char*>(buf.data()), k * sizeof(OctreeCube));
            for (size_t i = 0; i < k; ++i) {
                if (buf[i].depth < min_depth) min_depth = buf[i].depth;
                if (buf[i].depth > max_depth) max_depth = buf[i].depth;
            }
            remaining -= static_cast<uint32_t>(k);
        }
    }
    if (max_depth < min_depth) return;

    // Frozen-bars scratch file lives next to the state file.
    std::string bars_path = out_state_path + ".bars.tmp";

    for (int L = min_depth; L <= max_depth; ++L) {

        // -------- coarse-to-fine parent init for level L ------------------
        // For each leaf, find the cubes at depth L, look up their parent cubes
        // (anywhere in the file at depth < L), copy parent.u/v into them,
        // zero p/q.
        if (L > min_depth) {
            for (const auto& leaf : leaves) {
                auto leaf_cubes = load_cubes_range(balanced_cube_path,
                                                    leaf.first_idx, leaf.last_idx,
                                                    r_root_tt);
                std::vector<uint32_t> level_local;       // local indices into leaf_cubes
                std::vector<MortonCode> parent_codes;    // parent for each
                level_local.reserve(leaf_cubes.size());
                for (uint32_t li = 0; li < leaf_cubes.size(); ++li) {
                    if (leaf_cubes[li].depth == L) {
                        level_local.push_back(li);
                        parent_codes.push_back(parent_code(leaf_cubes[li].code, L));
                    }
                }
                if (level_local.empty()) continue;

                // Look up parents in the global cube file.
                std::vector<uint32_t> parent_global(level_local.size());
                {
                    CubeFileReader cr(balanced_cube_path);
                    for (size_t i = 0; i < parent_codes.size(); ++i)
                        parent_global[i] = cr.lookup(parent_codes[i]);
                }

                // Distinct parents → read parent state in one pass.
                std::vector<uint32_t> uniq_parents;
                uniq_parents.reserve(parent_global.size());
                for (uint32_t g : parent_global)
                    if (g != UINT32_MAX) uniq_parents.push_back(g);
                std::sort(uniq_parents.begin(), uniq_parents.end());
                uniq_parents.erase(std::unique(uniq_parents.begin(), uniq_parents.end()),
                                    uniq_parents.end());
                auto parent_states = load_tgv_at(out_state_path, uniq_parents);
                std::unordered_map<uint32_t, uint32_t> p_pos;
                p_pos.reserve(uniq_parents.size());
                for (uint32_t i = 0; i < uniq_parents.size(); ++i)
                    p_pos[uniq_parents[i]] = i;

                // Load this leaf's state, mutate level-L entries, save back.
                auto leaf_state = load_tgv_range(out_state_path,
                                                  leaf.first_idx, leaf.last_idx);
                for (size_t i = 0; i < level_local.size(); ++i) {
                    uint32_t li = level_local[i];
                    uint32_t pg = parent_global[i];
                    if (pg != UINT32_MAX) {
                        const TGVCube& ps = parent_states[p_pos[pg]];
                        leaf_state[li].u    = ps.u;
                        leaf_state[li].v[0] = ps.v[0];
                        leaf_state[li].v[1] = ps.v[1];
                        leaf_state[li].v[2] = ps.v[2];
                    }
                    leaf_state[li].p[0] = leaf_state[li].p[1] = leaf_state[li].p[2] = 0.0f;
                    for (int k = 0; k < 6; ++k) leaf_state[li].q[k] = 0.0f;
                }
                save_tgv_range(out_state_path, leaf.first_idx, leaf_state);
            }
        }

        // -------- snapshot frozen bars from state ------------------------
        build_bars_frozen(out_state_path, bars_path);

        // -------- iterate each leaf at this level ------------------------
        for (const auto& leaf : leaves) {
            auto leaf_cubes = load_cubes_range(balanced_cube_path,
                                                leaf.first_idx, leaf.last_idx,
                                                r_root_tt);
            const uint32_t n_leaf = static_cast<uint32_t>(leaf_cubes.size());

            // Active positions within the leaf for the current depth L.
            std::vector<uint32_t> active_pos;
            active_pos.reserve(n_leaf);
            for (uint32_t li = 0; li < n_leaf; ++li)
                if (leaf_cubes[li].depth == L) active_pos.push_back(li);
            if (active_pos.empty()) continue;

            // In-leaf CodeMap covers ALL leaf cubes (not just active depth),
            // so coarser parents available in the leaf serve as CTF neighbours.
            std::unordered_map<uint64_t, uint32_t> leaf_map;
            leaf_map.reserve(n_leaf);
            for (uint32_t li = 0; li < n_leaf; ++li)
                leaf_map[key(leaf_cubes[li].code)] = li;

            // Gather border neighbour codes for active cubes (CTF fallback).
            std::vector<MortonCode> border_codes;
            border_codes.reserve(active_pos.size() * 6);
            for (uint32_t li : active_pos) {
                const OctreeCube& c = leaf_cubes[li];
                for (int f = 0; f < 6; ++f) {
                    MortonCode nb = face_neighbor(c.code, f, c.depth);
                    if (nb.hi == UINT32_MAX) continue;
                    if (leaf_map.count(key(nb))) continue;
                    border_codes.push_back(nb);
                    if (c.depth > 0) {
                        MortonCode pnb = parent_code(nb, c.depth);
                        if (!leaf_map.count(key(pnb)))
                            border_codes.push_back(pnb);
                    }
                }
            }
            std::sort(border_codes.begin(), border_codes.end());
            border_codes.erase(std::unique(border_codes.begin(), border_codes.end()),
                                border_codes.end());

            // Look up each border code in the global cube file; keep those that exist
            // and fall OUTSIDE the leaf.
            std::vector<uint32_t>  border_global;
            std::vector<OctreeCube> border_cubes;
            border_global.reserve(border_codes.size());
            border_cubes.reserve(border_codes.size());
            {
                CubeFileReader cr(balanced_cube_path);
                for (const auto& code : border_codes) {
                    uint32_t g = cr.lookup(code);
                    if (g == UINT32_MAX) continue;
                    if (g >= leaf.first_idx && g <= leaf.last_idx) continue;
                    border_cubes.push_back(cr.read_at(g));
                    border_global.push_back(g);
                }
            }
            const uint32_t n_border = static_cast<uint32_t>(border_cubes.size());

            // Local indices: leaf cubes at [0, n_leaf), border at [n_leaf, n_leaf+n_border).
            std::unordered_map<uint64_t, uint32_t> local_map = std::move(leaf_map);
            local_map.reserve(n_leaf + n_border);
            for (uint32_t bi = 0; bi < n_border; ++bi)
                local_map[key(border_cubes[bi].code)] = n_leaf + bi;

            // Neighbour table for active cubes only.
            LocalNbt nbt;
            nbt.nb.assign(active_pos.size() * 6, -1);
            for (size_t ai = 0; ai < active_pos.size(); ++ai) {
                const OctreeCube& c = leaf_cubes[active_pos[ai]];
                for (int f = 0; f < 6; ++f) {
                    MortonCode nb_code = face_neighbor(c.code, f, c.depth);
                    if (nb_code.hi == UINT32_MAX) continue;
                    auto it = local_map.find(key(nb_code));
                    if (it != local_map.end()) { nbt.nb[ai*6+f] = (int32_t)it->second; continue; }
                    if (c.depth > 0) {
                        MortonCode par = parent_code(nb_code, c.depth);
                        it = local_map.find(key(par));
                        if (it != local_map.end()) nbt.nb[ai*6+f] = (int32_t)it->second;
                    }
                }
            }

            // Load leaf hists + state, plus border state and frozen bars.
            auto leaf_hists      = load_histograms_range(hist_path,
                                                          leaf.first_idx, leaf.last_idx);
            auto leaf_state_full = load_tgv_range(out_state_path,
                                                   leaf.first_idx, leaf.last_idx);

            std::vector<TGVCube> state_local(n_leaf + n_border);
            for (uint32_t li = 0; li < n_leaf; ++li)
                state_local[li] = leaf_state_full[li];
            auto border_state = load_tgv_at(out_state_path, border_global);
            for (uint32_t bi = 0; bi < n_border; ++bi) {
                state_local[n_leaf + bi] = border_state[bi];
                // Suppress cross-leaf coupling.
                for (int k = 0; k < 3; ++k) state_local[n_leaf + bi].p[k] = 0.0f;
                for (int k = 0; k < 6; ++k) state_local[n_leaf + bi].q[k] = 0.0f;
            }

            // Bars: in-leaf seeded from state, border seeded from frozen file.
            std::vector<float> u_bar (n_leaf + n_border),
                               vx_bar(n_leaf + n_border),
                               vy_bar(n_leaf + n_border),
                               vz_bar(n_leaf + n_border);
            for (uint32_t li = 0; li < n_leaf; ++li) {
                u_bar[li]  = state_local[li].u;
                vx_bar[li] = state_local[li].v[0];
                vy_bar[li] = state_local[li].v[1];
                vz_bar[li] = state_local[li].v[2];
            }
            auto border_bars = load_bars_at(bars_path, border_global);
            for (uint32_t bi = 0; bi < n_border; ++bi) {
                u_bar [n_leaf + bi] = border_bars[bi].u;
                vx_bar[n_leaf + bi] = border_bars[bi].vx;
                vy_bar[n_leaf + bi] = border_bars[bi].vy;
                vz_bar[n_leaf + bi] = border_bars[bi].vz;
            }

            run_level_streaming(active_pos, leaf_hists, state_local,
                                u_bar, vx_bar, vy_bar, vz_bar, nbt, params);

            // Write back leaf state (only active positions were mutated).
            for (uint32_t li = 0; li < n_leaf; ++li)
                leaf_state_full[li] = state_local[li];
            save_tgv_range(out_state_path, leaf.first_idx, leaf_state_full);
        }
    }

    std::remove(bars_path.c_str());
}
