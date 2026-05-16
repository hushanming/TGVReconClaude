cmake_minimum_required(VERSION 3.20)
project(TGVRecon CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(TBB REQUIRED)

set(LIBOMP_DIR "/opt/homebrew/Cellar/libomp/20.1.5")

add_library(tgvrecon_stage1 STATIC
    src/depth_map.cpp
    src/octree_builder.cpp)
target_include_directories(tgvrecon_stage1 PUBLIC src)
target_include_directories(tgvrecon_stage1 PRIVATE "${LIBOMP_DIR}/include")
target_link_libraries(tgvrecon_stage1 PUBLIC Eigen3::Eigen TBB::tbb)
target_link_libraries(tgvrecon_stage1 PRIVATE "${LIBOMP_DIR}/lib/libomp.dylib")
target_compile_options(tgvrecon_stage1 PRIVATE -Xpreprocessor -fopenmp)

add_library(tgvrecon STATIC
    src/depth_map.cpp
    src/octree_builder.cpp
    src/merge.cpp
    src/balance.cpp
    src/treetop.cpp
    src/histogram.cpp
    src/tgv.cpp
    src/marching_cubes.cpp
    src/qslim.cpp)
target_include_directories(tgvrecon PUBLIC src)
target_include_directories(tgvrecon PRIVATE "${LIBOMP_DIR}/include")
target_link_libraries(tgvrecon PUBLIC Eigen3::Eigen TBB::tbb)
target_link_libraries(tgvrecon PRIVATE "${LIBOMP_DIR}/lib/libomp.dylib")
target_compile_options(tgvrecon PRIVATE -Xpreprocessor -fopenmp)

add_executable(tgvrecon_main src/main.cpp)
target_link_libraries(tgvrecon_main PRIVATE tgvrecon)

add_executable(test_stage1 tests/test_stage1.cpp)
target_link_libraries(test_stage1 PRIVATE tgvrecon)

add_executable(test_stages2_4 tests/test_stages2_4.cpp)
target_link_libraries(test_stages2_4 PRIVATE tgvrecon)

add_executable(test_stage5 tests/test_stage5.cpp)
target_link_libraries(test_stage5 PRIVATE tgvrecon)

add_executable(test_stage6 tests/test_stage6.cpp)
target_link_libraries(test_stage6 PRIVATE tgvrecon)

add_executable(test_stage7 tests/test_stage7.cpp)
target_link_libraries(test_stage7 PRIVATE tgvrecon)
