import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_watermark import make_model


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser("Evaluate one watermark checkpoint on fixed cross-dataset subsets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--point_counts", type=parse_int_list, default=parse_int_list("4096,8192"))
    parser.add_argument("--datasets", type=parse_str_list, default=None)
    parser.add_argument("--alignment_stats", default=None)
    parser.add_argument("--alignment_source", default="objaverse")
    parser.add_argument("--alignment_target", default="modelnet")
    parser.add_argument(
        "--alignment_mode",
        choices=("none", "opacity", "scale", "rotation", "sh", "appearance", "all"),
        default="none",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    model = make_model(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_alignment(args):
    if args.alignment_mode == "none":
        return None
    if not args.alignment_stats:
        raise ValueError("--alignment_stats is required when alignment_mode is not none.")
    stats = json.loads(Path(args.alignment_stats).read_text())
    source = stats[args.alignment_source]["channels"]
    target = stats[args.alignment_target]["channels"]
    channel_names = (
        "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3", "sh_0", "sh_1", "sh_2",
    )
    groups = {
        "opacity": [3],
        "scale": [4, 5, 6],
        "rotation": [7, 8, 9, 10],
        "sh": [11, 12, 13],
        "appearance": list(range(3, 14)),
        "all": list(range(14)),
    }
    indices = groups[args.alignment_mode]
    return {
        "indices": indices,
        "source_mean": np.asarray([source[name]["mean"] for name in channel_names]),
        "source_std": np.asarray([source[name]["std"] for name in channel_names]),
        "target_mean": np.asarray([target[name]["mean"] for name in channel_names]),
        "target_std": np.asarray([target[name]["std"] for name in channel_names]),
        "target_min": np.asarray([target[name]["min"] for name in channel_names]),
        "target_max": np.asarray([target[name]["max"] for name in channel_names]),
    }


def align_points(points, alignment):
    if alignment is None:
        return points
    points = points.copy()
    indices = alignment["indices"]
    source_std = np.maximum(alignment["source_std"][indices], 1e-6)
    points[:, indices] = (
        (points[:, indices] - alignment["source_mean"][indices])
        / source_std
        * alignment["target_std"][indices]
        + alignment["target_mean"][indices]
    )
    points[:, indices] = np.clip(
        points[:, indices],
        alignment["target_min"][indices],
        alignment["target_max"][indices],
    )
    if any(index in indices for index in range(7, 11)):
        rotations = points[:, 7:11]
        rotations /= np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-9
        signs = np.sign(rotations[:, :1])
        signs[signs == 0] = 1
        points[:, 7:11] = rotations * signs
    return points


def evaluate_batch(model, points, messages, device):
    message = torch.from_numpy(messages).to(device=device, dtype=torch.float32)
    message = message * 2.0 - 1.0
    base = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    gaussians = base.unsqueeze(0).expand(message.shape[0], -1, -1)

    synchronize(device)
    start = time.perf_counter()
    embedder_output = model.embedder(gaussians, message)
    watermarked, delta = embedder_output[:2]
    synchronize(device)
    embed_ms = (time.perf_counter() - start) * 1000.0 / message.shape[0]

    start = time.perf_counter()
    logits = model.extractor(watermarked)
    synchronize(device)
    extract_ms = (time.perf_counter() - start) * 1000.0 / message.shape[0]

    correct = (logits >= 0) == (message > 0)
    gs_mse = torch.mean((watermarked - gaussians) ** 2, dim=(1, 2))
    xyz_mse = torch.mean((watermarked[..., :3] - gaussians[..., :3]) ** 2, dim=(1, 2))
    delta_abs = torch.mean(delta.abs(), dim=(1, 2))
    rows = []
    for index in range(message.shape[0]):
        rows.append(
            {
                "bit_acc": float(correct[index].float().mean().item()),
                "word_acc": float(correct[index].all().float().item()),
                "gs_mse": float(gs_mse[index].item()),
                "xyz_mse": float(xyz_mse[index].item()),
                "delta_abs": float(delta_abs[index].item()),
                "embed_ms": embed_ms,
                "extract_ms": extract_ms,
            }
        )
    return rows


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["point_count"])].append(record)

    metrics = ("bit_acc", "word_acc", "gs_mse", "xyz_mse", "delta_abs", "embed_ms", "extract_ms")
    results = []
    for (dataset, point_count), rows in sorted(groups.items()):
        result = {"dataset": dataset, "point_count": point_count, "num_records": len(rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            result[f"mean_{metric}"] = float(values.mean())
            result[f"std_{metric}"] = float(values.std())
        results.append(result)
    return results


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA because Gaussian-MAE grouping uses CUDA KNN.")
    device = torch.device("cuda:0")
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if any(count > int(manifest["max_points"]) for count in args.point_counts):
        raise ValueError("Requested point count exceeds cached point count.")

    model, checkpoint = load_model(args.checkpoint, device)
    alignment = build_alignment(args)
    expected_bits = int(checkpoint["args"]["num_bits"])
    if expected_bits != int(manifest["num_bits"]):
        raise ValueError("Checkpoint and cache payload lengths differ.")

    records = []
    records_path = output_dir / "metrics_per_asset.jsonl"
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "assets": len(manifest["assets"]),
                "point_counts": args.point_counts,
                "subsets_per_asset": manifest["subsets_per_asset"],
                "messages_per_asset": manifest["num_messages"],
            },
            indent=2,
        ),
        flush=True,
    )

    with records_path.open("w") as handle, torch.inference_mode():
        for asset_index, asset in enumerate(manifest["assets"]):
            if args.datasets and asset["dataset"] not in args.datasets:
                continue
            cached = np.load(cache_dir / asset["cache_path"], mmap_mode="r")
            messages = np.asarray(asset["messages"], dtype=np.float32)
            for point_count in args.point_counts:
                for subset_index in range(cached.shape[0]):
                    points = np.asarray(cached[subset_index, :point_count], dtype=np.float32)
                    if alignment is not None and asset["dataset"] == args.alignment_source:
                        points = align_points(points, alignment)
                    metrics = evaluate_batch(model, points, messages, device)
                    for message_index, row in enumerate(metrics):
                        record = {
                            "dataset": asset["dataset"],
                            "asset_index": asset["dataset_index"],
                            "source_path": asset["source_path"],
                            "source_points": asset["source_points"],
                            "point_count": point_count,
                            "subset_index": subset_index,
                            "message_index": message_index,
                            **row,
                        }
                        records.append(record)
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[asset {asset_index + 1}/{len(manifest['assets'])}] "
                f"{asset['dataset']}: {asset['source_path']}",
                flush=True,
            )

    summary = {
        "args": vars(args),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_assets": len(manifest["assets"]),
        "results": summarize(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
