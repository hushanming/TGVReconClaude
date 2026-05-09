#pragma once
#include <Eigen/Core>
#include <cstdint>

using Vec3f = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;

// 96-bit Morton code stored as three uint32 words representing the fully
// interleaved Z-curve code split into high/mid/low 32-bit words.
// Bits are interleaved x-y-z from LSB: bit0=x0, bit1=y0, bit2=z0, bit3=x1, ...
// Up to 10 bits per axis (30-bit interleaved code) fit in a single uint32;
// we extend to 3×32 = 96-bit by dilating 32 bits per axis across three words.
// For simplicity, we store dilated bits as uint64 per axis (21-bit support),
// then split into (hi, lo) words for the 96-bit representation.
//
// Practical layout: code.lo holds dilated bits 0..30, code.hi holds bits 31..62.
// Comparison on (hi, lo) gives correct Z-curve total order.
struct MortonCode {
    uint32_t hi = 0;  // upper 32 bits of interleaved code
    uint32_t lo = 0;  // lower 32 bits of interleaved code

    bool operator<(const MortonCode& o) const {
        if (hi != o.hi) return hi < o.hi;
        return lo < o.lo;
    }
    bool operator==(const MortonCode& o) const {
        return hi == o.hi && lo == o.lo;
    }
};

struct OctreeCube {
    MortonCode code;
    float r_c;       // avg sample radius (density)
    int32_t depth;
};

// Dilate a value into a 64-bit integer with 2 zero bits between each source bit.
// Supports up to 21 input bits (output occupies bits 0, 3, 6, ..., 60).
inline uint64_t dilate3_64(uint32_t v) {
    uint64_t x = v & 0x1fffffu;
    x = (x | (x << 32)) & 0x001f00000000ffffull;
    x = (x | (x << 16)) & 0x001f0000ff0000ffull;
    x = (x | (x <<  8)) & 0x100f00f00f00f00full;
    x = (x | (x <<  4)) & 0x10c30c30c30c30c3ull;
    x = (x | (x <<  2)) & 0x1249249249249249ull;
    return x;
}

// Compact a 64-bit dilated value back to 21-bit integer.
inline uint32_t compact3_64(uint64_t v) {
    v &= 0x1249249249249249ull;
    v = (v | (v >>  2)) & 0x10c30c30c30c30c3ull;
    v = (v | (v >>  4)) & 0x100f00f00f00f00full;
    v = (v | (v >>  8)) & 0x001f0000ff0000ffull;
    v = (v | (v >> 16)) & 0x001f00000000ffffull;
    v = (v | (v >> 32)) & 0x00000000001fffffull;
    return static_cast<uint32_t>(v);
}

// Encode integer grid cell (ix, iy, iz) each up to 21 bits to MortonCode.
inline MortonCode morton_encode(uint32_t ix, uint32_t iy, uint32_t iz) {
    uint64_t interleaved = dilate3_64(ix) | (dilate3_64(iy) << 1) | (dilate3_64(iz) << 2);
    return {static_cast<uint32_t>(interleaved >> 32),
            static_cast<uint32_t>(interleaved & 0xffffffffu)};
}

// Decode MortonCode back to grid cell indices.
inline void morton_decode(const MortonCode& m, uint32_t& ix, uint32_t& iy, uint32_t& iz) {
    uint64_t interleaved = (static_cast<uint64_t>(m.hi) << 32) | m.lo;
    ix = compact3_64(interleaved);
    iy = compact3_64(interleaved >> 1);
    iz = compact3_64(interleaved >> 2);
}
