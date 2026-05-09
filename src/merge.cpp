#include "merge.hpp"
#include "octree_builder.hpp"
#include <fstream>
#include <queue>
#include <stdexcept>

void kway_merge(const std::vector<std::string>& in_paths, const std::string& out_path) {
    if (in_paths.empty()) {
        // Write empty output with zeroed header.
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
    uint32_t total = 0;

    for (size_t i = 0; i < in_paths.size(); ++i) {
        streams[i].f.open(in_paths[i], std::ios::binary);
        if (!streams[i].f) throw std::runtime_error("Cannot open: " + in_paths[i]);
        CubeFileHeader hdr;
        streams[i].f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.magic[0]!='T'||hdr.magic[1]!='G'||hdr.magic[2]!='V'||hdr.magic[3]!='1')
            throw std::runtime_error("Bad magic: " + in_paths[i]);
        if (i == 0) r_root = hdr.r_root;
        total += hdr.cube_count;
        streams[i].advance();
    }

    // Min-heap: compare by Morton code.
    using Entry = std::pair<OctreeCube, int>;  // (cube, stream_index)
    auto cmp = [](const Entry& a, const Entry& b) { return b.first.code < a.first.code; };
    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);

    for (int i = 0; i < (int)streams.size(); ++i)
        if (streams[i].valid)
            heap.push({streams[i].cur, i});

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + out_path);

    CubeFileHeader out_hdr;
    out_hdr.r_root = r_root;
    out_hdr.cube_count = total;
    out.write(reinterpret_cast<const char*>(&out_hdr), sizeof(out_hdr));

    while (!heap.empty()) {
        auto [cube, idx] = heap.top();
        heap.pop();
        out.write(reinterpret_cast<const char*>(&cube), sizeof(OctreeCube));
        if (streams[idx].advance())
            heap.push({streams[idx].cur, idx});
    }
}
