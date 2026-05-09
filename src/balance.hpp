#pragma once
#include "types.hpp"
#include <string>

// Returns the Morton code of the parent (ancestor at depth-1).
// code: fine-grid Morton code of a cube at current_depth.
MortonCode parent_code(const MortonCode& code, int current_depth);

// Returns the fine-grid Morton code of the face-adjacent neighbor at the same depth.
// face: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
// Returns {UINT32_MAX, UINT32_MAX} when the neighbor is outside the scene.
MortonCode face_neighbor(const MortonCode& code, int face, int depth);

// Enforce 2:1 balance on a sorted OctreeCube file.
// Adds "bridge" cubes so no two adjacent leaves differ in depth by more than 1.
void balance_octree(const std::string& in_path, const std::string& out_path);
