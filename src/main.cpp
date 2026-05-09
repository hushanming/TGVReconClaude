#include "depth_map.hpp"
#include "octree_builder.hpp"
#include "merge.hpp"
#include "balance.hpp"
#include "treetop.hpp"
#include "histogram.hpp"
#include "tgv.hpp"
#include "marching_cubes.hpp"
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <depth_map_dir> <output_dir> <r_root> [ox oy oz]\n"
                 "  depth_map_dir: directory of .bin depth maps\n"
                 "  output_dir:    directory to write intermediate and final files\n"
                 "  r_root:        half-size of root octree cube (scene radius)\n"
                 "  ox oy oz:      scene origin (default 0 0 0)\n"
                 "\n"
                 "Depth map .bin format (row-major float32 array):\n"
                 "  [0]     width  (int32)\n"
                 "  [1]     height (int32)\n"
                 "  [2..10] K matrix row-major (9 float32)\n"
                 "  [11..26] Rt matrix row-major (16 float32)\n"
                 "  [27..]  width*height depth values (float32, NaN=invalid)\n";
}

static DepthMap load_bin_depthmap(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open depth map: " + path);

    int32_t w, h;
    f.read(reinterpret_cast<char*>(&w), sizeof(w));
    f.read(reinterpret_cast<char*>(&h), sizeof(h));

    DepthMap dm;
    dm.width  = w;
    dm.height = h;

    float kbuf[9], rtbuf[16];
    f.read(reinterpret_cast<char*>(kbuf),  sizeof(kbuf));
    f.read(reinterpret_cast<char*>(rtbuf), sizeof(rtbuf));

    dm.K  = Eigen::Map<Eigen::Matrix<float,3,3,Eigen::RowMajor>>(kbuf);
    dm.Rt = Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor>>(rtbuf);

    dm.depths.resize(w * h);
    f.read(reinterpret_cast<char*>(dm.depths.data()), w * h * sizeof(float));
    return dm;
}

int main(int argc, char* argv[]) {
    if (argc < 4) { usage(argv[0]); return 1; }

    std::string dm_dir  = argv[1];
    std::string out_dir = argv[2];
    float r_root = std::stof(argv[3]);
    Vec3f origin = Vec3f::Zero();
    if (argc >= 7)
        origin = {std::stof(argv[4]), std::stof(argv[5]), std::stof(argv[6])};

    if (!fs::is_directory(dm_dir)) {
        std::cerr << "Not a directory: " << dm_dir << "\n";
        return 1;
    }
    fs::create_directories(out_dir);

    // ---- Stage 1: depth maps → per-map .cubes files -------------------------
    std::cout << "[Stage 1] Building per-depthmap cube files...\n";
    fs::path stage1_dir = fs::path(out_dir) / "stage1";
    fs::create_directories(stage1_dir);

    std::vector<std::string> cube_paths;
    std::vector<DepthMap> raw_dms;

    for (const auto& entry : fs::directory_iterator(dm_dir)) {
        if (entry.path().extension() != ".bin") continue;
        std::string in_path  = entry.path().string();
        std::string out_path = (stage1_dir / entry.path().stem()).string() + ".cubes";
        try {
            DepthMap dm = load_bin_depthmap(in_path);
            build_cubes_from_depthmap(dm, origin, r_root, out_path);
            cube_paths.push_back(out_path);
            raw_dms.push_back(std::move(dm));
            std::cout << "  " << out_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "  skip " << in_path << ": " << e.what() << "\n";
        }
    }
    if (cube_paths.empty()) {
        std::cerr << "No depth maps processed.\n";
        return 1;
    }
    std::cout << "  " << cube_paths.size() << " depth maps processed.\n";

    // ---- Stage 2: k-way merge -----------------------------------------------
    std::cout << "[Stage 2] K-way merge...\n";
    std::string merged_path = (fs::path(out_dir) / "merged.cubes").string();
    kway_merge(cube_paths, merged_path);

    // ---- Stage 3: 2:1 balance -----------------------------------------------
    std::cout << "[Stage 3] Balancing octree...\n";
    std::string balanced_path = (fs::path(out_dir) / "balanced.cubes").string();
    balance_octree(merged_path, balanced_path);

    // ---- Stage 4: treetop index ---------------------------------------------
    std::cout << "[Stage 4] Building treetop index...\n";
    std::string treetop_path = (fs::path(out_dir) / "treetop.bin").string();
    build_treetop(balanced_path, treetop_path);

    float r_root_check;
    auto leaves = load_treetop(treetop_path, r_root_check);
    std::cout << "  " << leaves.size() << " treetop leaves.\n";

    // ---- Load balanced cubes for stages 5-7 ---------------------------------
    std::vector<OctreeCube> cubes = load_cubes(balanced_path, r_root_check);
    std::cout << "  " << cubes.size() << " balanced cubes.\n";

    std::vector<DepthMipmap> mipmaps;
    mipmaps.reserve(raw_dms.size());
    for (auto& dm : raw_dms) {
        DepthMipmap mm;
        mm.build(dm);
        mipmaps.push_back(std::move(mm));
    }

    // ---- Stage 5: histogram computation -------------------------------------
    std::cout << "[Stage 5] Computing histograms...\n";
    auto hists = compute_histograms(cubes, mipmaps, origin, r_root);
    std::string hist_path = (fs::path(out_dir) / "histograms.bin").string();
    save_histograms(hist_path, hists);

    // ---- Stage 6: TGV minimization ------------------------------------------
    std::cout << "[Stage 6] TGV minimization...\n";
    TGVParams params;
    auto tgv_state = tgv_minimize(cubes, hists, origin, r_root, params);
    std::string tgv_path = (fs::path(out_dir) / "tgv_state.bin").string();
    save_tgv(tgv_path, tgv_state);

    // ---- Stage 7: marching cubes + write mesh --------------------------------
    std::cout << "[Stage 7] Extracting surface...\n";
    std::vector<bool> has_data(cubes.size());
    for (size_t i = 0; i < hists.size(); ++i)
        has_data[i] = hists[i].total() > 0;
    auto tris = extract_surface(cubes, tgv_state, origin, r_root, has_data);
    std::string mesh_path = (fs::path(out_dir) / "mesh.obj").string();
    write_obj(mesh_path, tris);
    std::cout << "  " << tris.size() << " triangles → " << mesh_path << "\n";

    std::cout << "Done.\n";
    return 0;
}
