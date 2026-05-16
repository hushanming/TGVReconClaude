#pragma once
#include "marching_cubes.hpp"
#include <cstddef>

// QSlim quadric-error metric edge-collapse decimation.
//
// Each vertex carries a 4×4 quadric Q (Garland & Heckbert 1997).
// Iteratively collapses the cheapest edge (smallest quadric cost) until
// tri_count ≤ target_faces or no more collapsible edges remain.
//
// border_vertex[i] = true marks vertex i as lying on a treetop-leaf cube face
// boundary; edges where BOTH endpoints are border vertices are never collapsed
// (preserves seamless leaf joins as required by the paper).
Mesh decimate(const Mesh& mesh,
              size_t target_faces,
              const std::vector<bool>& border_vertex = {});
