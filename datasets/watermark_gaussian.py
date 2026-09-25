from pathlib import Path

import math
import numpy as np
import torch
import torch.utils.data as data
from plyfile import PlyData


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_gaussian_ply(path, attribute):
    vertex = PlyData.read(path)["vertex"]
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    out = [np.stack((x, y, z), axis=-1)]

    if "opacity" in attribute:
        out.append(_sigmoid(vertex["opacity"].astype(np.float32)).reshape(-1, 1))

    if "scale" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda s: int(s.split("_")[-1]))
        scales = np.stack([vertex[name].astype(np.float32) for name in scale_names], axis=-1)
        out.append(np.exp(scales))

    if "rotation" in attribute:
        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda s: int(s.split("_")[-1]))
        rots = np.stack([vertex[name].astype(np.float32) for name in rot_names], axis=-1)
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        signs = np.sign(rots[:, :1])
        signs[signs == 0] = 1
        out.append(rots * signs)

    if "sh" in attribute:
        sh = np.stack(
            [
                vertex["f_dc_0"].astype(np.float32),
                vertex["f_dc_1"].astype(np.float32),
                vertex["f_dc_2"].astype(np.float32),
            ],
            axis=-1,
        )
        out.append(sh)

    return np.concatenate(out, axis=-1)


def normalize_gaussian(data, norm_attribute):
    data = data.copy()
    xyz = data[:, :3]
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    radius = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    radius = max(float(radius), 1e-6)
    data[:, :3] = xyz / radius

    if data.shape[1] >= 7:
        data[:, 4:7] = data[:, 4:7] / radius

    if "opacity" in norm_attribute and data.shape[1] >= 4:
        data[:, 3] = data[:, 3] * 2.0 - 1.0

    if "scale" in norm_attribute and data.shape[1] >= 7:
        scale = data[:, 4:7]
        scale_center = scale.mean(axis=0)
        scale = scale - scale_center
        scale_radius = np.max(np.sqrt(np.sum(scale**2, axis=1)))
        scale_radius = max(float(scale_radius), 1e-6)
        data[:, 4:7] = scale / scale_radius

    if "sh" in norm_attribute and data.shape[1] >= 14:
        sh = data[:, 11:14] * 0.28209479177387814
        sh = np.clip(sh, -0.5, 0.5)
        data[:, 11:14] = 2.0 * sh / math.sqrt(3.0)

    return data


def gaussian_normalization_stats(data, norm_attribute):
    """Return the full-asset statistics needed to invert normalize_gaussian."""
    xyz = data[:, :3]
    centroid = xyz.mean(axis=0)
    centered_xyz = xyz - centroid
    radius = max(float(np.max(np.linalg.norm(centered_xyz, axis=1))), 1e-6)
    stats = {"xyz_centroid": centroid, "xyz_radius": radius}

    if "scale" in norm_attribute and data.shape[1] >= 7:
        scaled = data[:, 4:7] / radius
        scale_center = scaled.mean(axis=0)
        scale_radius = max(
            float(np.max(np.linalg.norm(scaled - scale_center, axis=1))), 1e-6
        )
        stats.update(scale_center=scale_center, scale_radius=scale_radius)
    return stats


def denormalize_gaussian(data, norm_attribute, stats):
    """Invert normalize_gaussian into read_gaussian_ply's physical parameterization."""
    data = data.copy()
    radius = float(stats["xyz_radius"])
    data[:, :3] = data[:, :3] * radius + stats["xyz_centroid"]

    if "opacity" in norm_attribute and data.shape[1] >= 4:
        data[:, 3] = np.clip((data[:, 3] + 1.0) * 0.5, 1e-6, 1.0 - 1e-6)

    if "scale" in norm_attribute and data.shape[1] >= 7:
        scaled = data[:, 4:7] * float(stats["scale_radius"]) + stats["scale_center"]
        data[:, 4:7] = np.maximum(scaled * radius, 1e-8)

    if "rotation" in norm_attribute and data.shape[1] >= 11:
        rotations = data[:, 7:11]
        rotations /= np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-9
        signs = np.sign(rotations[:, :1])
        signs[signs == 0] = 1
        data[:, 7:11] = rotations * signs

    if "sh" in norm_attribute and data.shape[1] >= 14:
        data[:, 11:14] = (
            data[:, 11:14]
            * math.sqrt(3.0)
            / (2.0 * 0.28209479177387814)
        )
    return data


class WatermarkGaussianDataset(data.Dataset):
    def __init__(
        self,
        roots,
        npoints=4096,
        attribute=("xyz", "opacity", "scale", "rotation", "sh"),
        norm_attribute=("xyz", "opacity", "scale", "rotation", "sh"),
        max_files=None,
        file_list=None,
        deterministic=False,
        seed=0,
    ):
        self.roots = [Path(root) for root in roots if root]
        self.npoints = npoints
        self.attribute = tuple(attribute)
        self.norm_attribute = tuple(norm_attribute)
        self.deterministic = deterministic
        self.seed = seed
        self.files = []

        if file_list:
            self.files = [line.strip() for line in Path(file_list).read_text().splitlines() if line.strip()]
        else:
            for root in self.roots:
                if not root.exists():
                    continue
                self.files.extend(self._scan_root(root))

        self.files = sorted(set(self.files))
        if max_files is not None and max_files > 0:
            self.files = self.files[:max_files]

        if not self.files:
            searched = ", ".join(str(r) for r in self.roots)
            raise RuntimeError(f"No Gaussian ply files found under: {searched}")

    def _scan_root(self, root):
        modelnet = list(root.glob("*/*/*/point_cloud.ply"))
        objaverse = list(root.glob("*/*/ckpts/point_cloud_15000.ply"))
        flat = list(root.glob("*.ply"))
        return [str(p) for p in modelnet + objaverse + flat]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        gs = read_gaussian_ply(path, self.attribute)
        gs = normalize_gaussian(gs, self.norm_attribute)
        replace = len(gs) < self.npoints
        rng = np.random.default_rng(self.seed + idx) if self.deterministic else np.random
        choice = rng.choice(len(gs), self.npoints, replace=replace)
        if not replace:
            rng.shuffle(choice)
        gs = torch.from_numpy(gs[choice]).float()
        return path, gs
