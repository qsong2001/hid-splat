import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from datasets.watermark_gaussian import WatermarkGaussianDataset, normalize_gaussian, read_gaussian_ply


ATTRIBUTES = ("xyz", "opacity", "scale", "rotation", "sh")
EXPECTED_DATASET_SIZES = {
    "modelnet": 12309,
    "shapenet": 52121,
    "objaverse": 141703,
}


def parse_args():
    parser = argparse.ArgumentParser("Prepare fixed 4K/8K watermark evaluation subsets")
    parser.add_argument("--modelnet_root", default="gs_data/modelsplat/modelsplat_ply")
    parser.add_argument("--shapenet_root", default="gs_data/shapesplat/shapesplat_ply")
    parser.add_argument("--objaverse_root", default="gs_data/objaverse_splats/objaverse_ply")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--assets_per_dataset", type=int, default=100)
    parser.add_argument("--subsets_per_asset", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=8192)
    parser.add_argument("--num_messages", type=int, default=5)
    parser.add_argument("--num_bits", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--modelnet_split_seed", type=int, default=0)
    parser.add_argument("--modelnet_val_ratio", type=float, default=0.02)
    parser.add_argument("--datasets", default="modelnet,shapenet,objaverse")
    parser.add_argument("--modelnet_file_list")
    parser.add_argument("--shapenet_file_list")
    parser.add_argument("--objaverse_file_list")
    parser.add_argument("--file_list_root", default=".")
    return parser.parse_args()


def scan_files(root):
    dataset = WatermarkGaussianDataset(roots=[root], npoints=1)
    return dataset.files


def select_modelnet(files, count, split_seed, val_ratio):
    val_len = max(1, int(len(files) * val_ratio))
    train_len = len(files) - val_len
    _, val_set = random_split(
        range(len(files)),
        [train_len, val_len],
        generator=torch.Generator().manual_seed(split_seed),
    )
    if len(val_set.indices) < count:
        raise RuntimeError(f"ModelNet held-out split has only {len(val_set.indices)} assets, requested {count}.")
    return [files[index] for index in val_set.indices[:count]]


def select_zero_shot(files, count, seed):
    if len(files) < count:
        raise RuntimeError(f"Dataset has only {len(files)} assets, requested {count}.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=count, replace=False)
    return [files[int(index)] for index in indices]


def read_file_list(path, root, count):
    entries = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(entries) != count:
        raise RuntimeError(f"Expected exactly {count} paths in {path}, found {len(entries)}")
    resolved = []
    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = Path(root) / candidate
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resolved.append(str(candidate))
    return resolved


def cache_asset(path, output_path, subsets, max_points, seed):
    raw = read_gaussian_ply(path, ATTRIBUTES)
    normalized = normalize_gaussian(raw, ATTRIBUTES)
    rng = np.random.default_rng(seed)
    cached = np.empty((subsets, max_points, normalized.shape[1]), dtype=np.float32)
    for subset_index in range(subsets):
        replace = len(normalized) < max_points
        indices = rng.choice(len(normalized), size=max_points, replace=replace)
        cached[subset_index] = normalized[indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, cached)
    return len(normalized)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")

    requested_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    roots_all = {
        "modelnet": args.modelnet_root,
        "shapenet": args.shapenet_root,
        "objaverse": args.objaverse_root,
    }
    unknown = sorted(set(requested_datasets) - set(roots_all))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    roots = {name: roots_all[name] for name in requested_datasets}
    file_lists = {
        "modelnet": args.modelnet_file_list,
        "shapenet": args.shapenet_file_list,
        "objaverse": args.objaverse_file_list,
    }
    files = {}
    for name, root in roots.items():
        if file_lists[name]:
            files[name] = read_file_list(
                file_lists[name], args.file_list_root, args.assets_per_dataset
            )
        else:
            files[name] = scan_files(root)
            expected_size = EXPECTED_DATASET_SIZES[name]
            if len(files[name]) != expected_size:
                raise RuntimeError(
                    f"{name} contains {len(files[name])} readable PLY paths; expected {expected_size}. "
                    "Dataset download or extraction is incomplete."
                )
    selected = {}
    if "modelnet" in requested_datasets:
        selected["modelnet"] = files["modelnet"] if file_lists["modelnet"] else select_modelnet(
            files["modelnet"], args.assets_per_dataset, args.modelnet_split_seed, args.modelnet_val_ratio
        )
    if "shapenet" in requested_datasets:
        selected["shapenet"] = files["shapenet"] if file_lists["shapenet"] else select_zero_shot(files["shapenet"], args.assets_per_dataset, args.seed + 1)
    if "objaverse" in requested_datasets:
        selected["objaverse"] = files["objaverse"] if file_lists["objaverse"] else select_zero_shot(files["objaverse"], args.assets_per_dataset, args.seed + 2)

    message_rng = np.random.default_rng(args.seed + 3)
    manifest = {
        "protocol": "paired_sampled_subsets",
        "seed": args.seed,
        "num_bits": args.num_bits,
        "num_messages": args.num_messages,
        "subsets_per_asset": args.subsets_per_asset,
        "max_points": args.max_points,
        "paired_point_counts": [4096, args.max_points],
        "datasets": requested_datasets,
        "dataset_sizes": {name: len(paths) for name, paths in files.items()},
        "assets": [],
    }

    total = sum(len(paths) for paths in selected.values())
    completed = 0
    for dataset_name, paths in selected.items():
        for dataset_index, path in enumerate(paths):
            cache_relpath = Path(dataset_name) / f"{dataset_index:04d}.npy"
            source_points = cache_asset(
                path,
                output_dir / cache_relpath,
                args.subsets_per_asset,
                args.max_points,
                args.seed + completed * 1009,
            )
            messages = message_rng.integers(
                0,
                2,
                size=(args.num_messages, args.num_bits),
                dtype=np.int8,
            )
            manifest["assets"].append(
                {
                    "dataset": dataset_name,
                    "dataset_index": dataset_index,
                    "source_path": path,
                    "source_points": source_points,
                    "cache_path": str(cache_relpath),
                    "messages": messages.astype(int).tolist(),
                }
            )
            completed += 1
            print(f"[{completed}/{total}] {dataset_name}: {path}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "assets": total}, indent=2), flush=True)


if __name__ == "__main__":
    main()
