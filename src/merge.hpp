#pragma once
#include "types.hpp"
#include <string>
#include <vector>

// K-way merge of sorted OctreeCube files into one sorted output file.
// All inputs must share the same r_root.
void kway_merge(const std::vector<std::string>& in_paths, const std::string& out_path);
