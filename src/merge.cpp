#include "merge.hpp"
#include "octree_builder.hpp"
#include <fstream>
#include <queue>
#include <stdexcept>

void kway_merge(const std::vector<std::string>& in_paths, const std::string& out_path) {
    if (in_paths.empty()) {
        std::ofstream out(out_path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open output: " + out_path);
        CubeFileHeader hdr;
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        return;
    }

    struct Stream {
        std::ifstream f;
        OctreeCube cur;
        bool valid = false;
        bool advance() {
            f.read(reinterpret_cast<char*>(&cur), sizeof(OctreeCube));
            valid = f.gcount() == sizeof(OctreeCube);
            return valid;
        }
    };

    std::vector<Stream> streams(in_paths.size());
    float r_root = 0.0f;

    for (size_t i = 0; i < in_paths.size(); ++i) {
        streams[i].f.open(in_paths[i], std::ios::binary);
        if (!streams[i].f) throw std::runtime_error("Cannot open: " + in_paths[i]);
        CubeFileHeader hdr;
        streams[i].f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='1')
            throw std::runtime_error("Bad magic: " + in_paths[i]);
        if (i == 0) r_root = hdr.r_root;
        streams[i].advance();
    }

    using Entry = std::pair<OctreeCube, int>;
    auto cmp = [](const Entry& a, const Entry& b) { return b.first.code < a.first.code; };
    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);

    for (int i = 0; i < static_cast<int>(streams.size()); ++i)
        if (streams[i].valid)
            heap.push({streams[i].cur, i});

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + out_path);

    // Write header placeholder; we will seek back to fill cube_count.
    CubeFileHeader out_hdr;
    out_hdr.r_root = r_root;
    out.write(reinterpret_cast<const char*>(&out_hdr), sizeof(out_hdr));

    // Merge with deduplication: cubes with same (code, depth) are merged and
    // their r_c values averaged.
    uint32_t cube_count = 0;
    bool has_pending = false;
    OctreeCube pending{};
    float pending_r_sum = 0.0f;
    uint32_t pending_r_cnt = 0;

    auto flush_pending = [&]() {
        if (!has_pending) return;
        pending.r_c = pending_r_sum / static_cast<float>(pending_r_cnt);
        out.write(reinterpret_cast<const char*>(&pending), sizeof(OctreeCube));
        ++cube_count;
        has_pending = false;
    };

    while (!heap.empty()) {
        auto [cube, idx] = heap.top();
        heap.pop();

        if (has_pending && cube.code == pending.code && cube.depth == pending.depth) {
            // Same cell: accumulate r_c.
            pending_r_sum += cube.r_c;
            ++pending_r_cnt;
        } else {
            flush_pending();
            pending       = cube;
            pending_r_sum = cube.r_c;
            pending_r_cnt = 1;
            has_pending   = true;
        }

        if (streams[idx].advance())
            heap.push({streams[idx].cur, idx});
    }
    flush_pending();

    // Patch the cube_count in the header.
    out.seekp(offsetof(CubeFileHeader, cube_count));
    out.write(reinterpret_cast<const char*>(&cube_count), sizeof(cube_count));
}
