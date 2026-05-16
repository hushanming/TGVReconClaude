#!/usr/bin/env python3
"""
Convert ETH3D dataset (COLMAP format + raw float32 depth maps) to TGVRecon .bin format.

Usage:
    python3 convert_eth3d.py <calibration_dir> <depth_dir> <output_dir>

Where:
    calibration_dir: contains cameras.txt and images.txt (COLMAP undistorted format)
    depth_dir:       contains raw float32 depth maps (original resolution)
    output_dir:      where to write .bin depth map files for tgvrecon_main
"""
import sys
import os
import struct
import numpy as np
from pathlib import Path


def quat_to_rotation(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[0, 1] = 2*(qx*qy - qz*qw)
    R[0, 2] = 2*(qx*qz + qy*qw)
    R[1, 0] = 2*(qx*qy + qz*qw)
    R[1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[1, 2] = 2*(qy*qz - qx*qw)
    R[2, 0] = 2*(qx*qz - qy*qw)
    R[2, 1] = 2*(qy*qz + qx*qw)
    R[2, 2] = 1 - 2*(qx*qx + qy*qy)
    return R


def parse_cameras(path):
    """Parse COLMAP cameras.txt. Returns dict: cam_id -> (model, w, h, params)."""
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = (model, w, h, params)
    return cameras


def parse_images(path):
    """Parse COLMAP images.txt. Returns list of (name, qw,qx,qy,qz, tx,ty,tz, cam_id)."""
    images = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    # Every other line is image data (odd lines are 2D points)
    for i in range(0, len(lines), 2):
        parts = lines[i].strip().split()
        if len(parts) < 10:
            continue
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        images.append((name, qw, qx, qy, qz, tx, ty, tz, cam_id))
    return images


def write_bin_depthmap(out_path, width, height, K, Rt, depths):
    """
    Write depth map in TGVRecon .bin format:
      int32 width, int32 height
      float32[9] K (row-major 3x3)
      float32[16] Rt (row-major 4x4)
      float32[width*height] depths (NaN = invalid)
    """
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<ii', width, height))
        f.write(K.astype(np.float32).tobytes())
        f.write(Rt.astype(np.float32).tobytes())
        f.write(depths.astype(np.float32).tobytes())


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    calib_dir = Path(sys.argv[1])
    depth_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    max_dim = int(sys.argv[4]) if len(sys.argv) >= 5 else 0  # 0 = no downscale
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = parse_cameras(calib_dir / 'cameras.txt')
    images = parse_images(calib_dir / 'images.txt')

    print(f"Cameras: {len(cameras)}, Images: {len(images)}, max_dim: {max_dim or 'none'}")

    converted = 0
    for name, qw, qx, qy, qz, tx, ty, tz, cam_id in images:
        img_basename = Path(name).name
        depth_path = depth_dir / img_basename
        if not depth_path.exists():
            print(f"  skip {img_basename}: no depth map")
            continue

        model, cam_w, cam_h, params = cameras[cam_id]
        if model != 'PINHOLE':
            print(f"  skip {img_basename}: unsupported model {model}")
            continue

        fx, fy, cx, cy = params[0], params[1], params[2], params[3]

        raw = np.fromfile(str(depth_path), dtype=np.float32)
        for dw, dh in [(6048, 4032), (6032, 4032), (cam_w, cam_h)]:
            if dw * dh == len(raw):
                break
        else:
            print(f"  skip {img_basename}: cannot determine depth dims ({len(raw)} pixels)")
            continue

        depth_map = raw.reshape(dh, dw)

        # Scale intrinsics from undistorted resolution to depth map resolution
        scale_x = dw / cam_w
        scale_y = dh / cam_h
        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = fx * scale_x
        K[1, 1] = fy * scale_y
        K[0, 2] = cx * scale_x
        K[1, 2] = cy * scale_y
        K[2, 2] = 1.0

        # Downsample if requested
        if max_dim > 0 and max(dw, dh) > max_dim:
            factor = max(dw, dh) / max_dim
            new_w = int(round(dw / factor))
            new_h = int(round(dh / factor))
            # Block-average downsample (preserves depth semantics better than interp)
            bw = dw // new_w
            bh = dh // new_h
            crop_w = bw * new_w
            crop_h = bh * new_h
            cropped = depth_map[:crop_h, :crop_w]
            # Replace invalid with NaN before averaging
            cropped = np.where(np.isfinite(cropped) & (cropped > 0), cropped, np.nan)
            blocks = cropped.reshape(new_h, bh, new_w, bw)
            with np.errstate(all='ignore'):
                depth_map = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)
            # Scale intrinsics
            K[0, 0] /= factor
            K[1, 1] /= factor
            K[0, 2] /= factor
            K[1, 2] /= factor
            dw, dh = new_w, new_h

        # Build 4x4 Rt (world-to-camera)
        R = quat_to_rotation(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float32)
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = R
        Rt[:3, 3] = t

        # Replace inf/negative with NaN
        invalid = ~np.isfinite(depth_map) | (depth_map <= 0)
        depth_map[invalid] = np.nan

        out_path = out_dir / (Path(img_basename).stem + '.bin')
        write_bin_depthmap(str(out_path), dw, dh, K, Rt, depth_map)
        valid_pct = 100 * np.sum(~invalid) / depth_map.size
        print(f"  {out_path.name}: {dw}x{dh}, {valid_pct:.1f}% valid, depth [{np.nanmin(depth_map):.2f}, {np.nanmax(depth_map):.2f}]")
        converted += 1

    print(f"\nConverted {converted} depth maps to {out_dir}")


if __name__ == '__main__':
    main()
