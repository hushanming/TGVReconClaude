#include "qslim.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <vector>

// ---------- quadric ----------------------------------------------------------

// 4×4 symmetric quadric stored as upper triangle (10 floats).
// Q(v) = v^T M v  for homogeneous v=(x,y,z,1)^T.
struct Quadric {
    float q[10] = {};  // q00,q01,q02,q03,q11,q12,q13,q22,q23,q33

    Quadric& operator+=(const Quadric& o) {
        for (int i = 0; i < 10; ++i) q[i] += o.q[i];
        return *this;
    }

    float eval(const Vec3f& p) const {
        float x=p.x(), y=p.y(), z=p.z();
        return q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x
             + q[4]*y*y + 2*q[5]*y*z + 2*q[6]*y
             + q[7]*z*z + 2*q[8]*z
             + q[9];
    }
};

static Quadric plane_quadric(const Vec3f& n, float d) {
    float a=n.x(), b=n.y(), c=n.z();
    Quadric kp;
    kp.q[0]=a*a; kp.q[1]=a*b; kp.q[2]=a*c; kp.q[3]=a*d;
    kp.q[4]=b*b; kp.q[5]=b*c; kp.q[6]=b*d;
    kp.q[7]=c*c; kp.q[8]=c*d;
    kp.q[9]=d*d;
    return kp;
}

// ---------- collapse candidate -----------------------------------------------

struct Candidate {
    float    cost;
    uint32_t v0, v1;
    Vec3f    target;
    bool operator>(const Candidate& o) const { return cost > o.cost; }
};

// ---------- main decimation --------------------------------------------------

Mesh decimate(const Mesh& in, size_t target_faces, const std::vector<bool>& border_vertex) {
    if (in.empty() || in.tri_count() <= target_faces) return in;

    const uint32_t NV = static_cast<uint32_t>(in.verts.size());
    const size_t   NT = in.tri_count();

    // Mutable working copies.
    std::vector<Vec3f>    verts = in.verts;
    std::vector<uint32_t> faces = in.faces;   // mutable so remap can update

    // ---- adjacency ----------------------------------------------------------
    std::vector<std::unordered_set<uint32_t>> adj(NV);
    std::vector<std::vector<uint32_t>>        vtri(NV);

    for (size_t t = 0; t < NT; ++t)
        for (int k = 0; k < 3; ++k) {
            uint32_t a = faces[t*3+k], b = faces[t*3+(k+1)%3];
            adj[a].insert(b); adj[b].insert(a);
            vtri[a].push_back(static_cast<uint32_t>(t));
        }

    // ---- per-vertex quadrics ------------------------------------------------
    std::vector<Quadric> Q(NV);
    for (size_t t = 0; t < NT; ++t) {
        uint32_t a = faces[t*3], b = faces[t*3+1], c = faces[t*3+2];
        Vec3f n = (verts[b]-verts[a]).cross(verts[c]-verts[a]);
        float nl = n.norm();
        if (nl < 1e-12f) continue;
        n /= nl;
        float d = -n.dot(verts[a]);
        Quadric kp = plane_quadric(n, d);
        Q[a] += kp; Q[b] += kp; Q[c] += kp;
    }

    // ---- state --------------------------------------------------------------
    std::vector<int32_t> remap(NV);
    std::iota(remap.begin(), remap.end(), 0);
    std::vector<bool> dead_tri(NT, false);
    std::vector<bool> dead_vert(NV, false);
    size_t live_tris = NT;

    auto canonical = [&](uint32_t v) -> uint32_t {
        while (remap[v] != static_cast<int32_t>(v)) v = static_cast<uint32_t>(remap[v]);
        return v;
    };

    // ---- compute collapse candidate for an edge ----------------------------
    auto make_cand = [&](uint32_t v0, uint32_t v1) -> Candidate {
        Quadric Qe = Q[v0]; Qe += Q[v1];
        Eigen::Matrix3f A;
        A << Qe.q[0], Qe.q[1], Qe.q[2],
             Qe.q[1], Qe.q[4], Qe.q[5],
             Qe.q[2], Qe.q[5], Qe.q[7];
        Eigen::Vector3f rhs(-Qe.q[3], -Qe.q[6], -Qe.q[8]);
        Vec3f target;
        if (std::abs(A.determinant()) > 1e-10f) {
            auto sol = A.ldlt().solve(rhs);
            target = Vec3f(sol.x(), sol.y(), sol.z());
        } else {
            target = 0.5f * (verts[v0] + verts[v1]);
        }
        return {Qe.eval(target), v0, v1, target};
    };

    // ---- priority queue -----------------------------------------------------
    std::priority_queue<Candidate,
                        std::vector<Candidate>,
                        std::greater<Candidate>> pq;

    for (uint32_t v = 0; v < NV; ++v)
        for (uint32_t nb : adj[v])
            if (nb > v)
                pq.push(make_cand(v, nb));

    // ---- collapse loop ------------------------------------------------------
    while (live_tris > target_faces && !pq.empty()) {
        Candidate cand = pq.top(); pq.pop();

        uint32_t v0 = canonical(cand.v0);
        uint32_t v1 = canonical(cand.v1);
        if (v0 == v1 || dead_vert[v0] || dead_vert[v1]) continue;

        // Border constraint: never collapse edges where both endpoints are border.
        if (!border_vertex.empty() && border_vertex[v0] && border_vertex[v1])
            continue;

        // Staleness: re-evaluate cost at stored target; skip if too stale.
        Quadric Qe = Q[v0]; Qe += Q[v1];
        float reeval = Qe.eval(cand.target);
        if (reeval > cand.cost * 8.0f + 1e-6f) {
            pq.push(make_cand(v0, v1));
            continue;
        }

        // Link condition: the set of vertices adjacent to BOTH v0 and v1
        // must equal the set of third vertices in triangles sharing this edge.
        // Without this check QSlim can create non-manifold geometry.
        {
            size_t shared_cnt = 0;
            for (uint32_t nb : adj[v0]) {
                uint32_t cnb = canonical(nb);
                if (cnb == v0 || cnb == v1) continue;
                bool found = false;
                for (uint32_t nb1 : adj[v1]) {
                    if (canonical(nb1) == cnb) { found = true; break; }
                }
                if (found) ++shared_cnt;
            }
            size_t edge_tris = 0;
            for (uint32_t ti : vtri[v0]) {
                if (dead_tri[ti]) continue;
                bool hv0=false, hv1=false;
                for (int k=0;k<3;++k) {
                    uint32_t ci = canonical(faces[ti*3+k]);
                    if (ci==v0) hv0=true; if (ci==v1) hv1=true;
                }
                if (hv0 && hv1) ++edge_tris;
            }
            if (shared_cnt != edge_tris) continue;
        }

        // Collapse v1 → v0.
        verts[v0] = cand.target;
        Q[v0] += Q[v1];
        remap[v1] = static_cast<int32_t>(v0);
        dead_vert[v1] = true;

        // Transfer v1's adjacency to v0.
        for (uint32_t nb : adj[v1]) {
            uint32_t cnb = canonical(nb);
            if (cnb == v0) continue;
            adj[v0].insert(cnb);
        }
        adj[v0].erase(v1);
        adj[v1].clear();

        // Update triangles touching v1.
        for (uint32_t ti : vtri[v1]) {
            if (dead_tri[ti]) continue;
            // Remap v1 → v0.
            for (int k = 0; k < 3; ++k)
                if (faces[ti*3+k] == v1) faces[ti*3+k] = v0;
            uint32_t a=canonical(faces[ti*3]), b=canonical(faces[ti*3+1]), c=canonical(faces[ti*3+2]);
            if (a==b || b==c || a==c) {
                dead_tri[ti] = true; --live_tris;
            } else {
                vtri[v0].push_back(ti);
            }
        }

        // Enqueue new collapse candidates for v0's neighbours.
        for (uint32_t nb : adj[v0]) {
            uint32_t cnb = canonical(nb);
            if (cnb == v0 || dead_vert[cnb]) continue;
            pq.push(make_cand(v0, cnb));
        }
    }

    // ---- compact output -----------------------------------------------------
    // Assign new indices to surviving vertices.
    std::vector<uint32_t> new_idx(NV, UINT32_MAX);
    Mesh out;
    for (uint32_t v = 0; v < NV; ++v) {
        if (dead_vert[v]) continue;
        uint32_t cv = canonical(v);
        if (cv == v && new_idx[v] == UINT32_MAX) {
            new_idx[v] = static_cast<uint32_t>(out.verts.size());
            out.verts.push_back(verts[v]);
        }
    }
    // Remapped vertices point to their canonical's new index.
    for (uint32_t v = 0; v < NV; ++v) {
        if (dead_vert[v] || new_idx[v] != UINT32_MAX) continue;
        uint32_t cv = canonical(v);
        new_idx[v] = new_idx[cv];
    }

    out.faces.reserve(live_tris * 3);
    for (size_t t = 0; t < NT; ++t) {
        if (dead_tri[t]) continue;
        uint32_t a = canonical(faces[t*3]);
        uint32_t b = canonical(faces[t*3+1]);
        uint32_t c = canonical(faces[t*3+2]);
        if (a==b||b==c||a==c) continue;
        if (new_idx[a]==UINT32_MAX||new_idx[b]==UINT32_MAX||new_idx[c]==UINT32_MAX) continue;
        out.faces.push_back(new_idx[a]);
        out.faces.push_back(new_idx[b]);
        out.faces.push_back(new_idx[c]);
    }

    return out;
}
