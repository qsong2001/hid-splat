import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from datasets.watermark_gaussian import WatermarkGaussianDataset, normalize_gaussian, read_gaussian_ply
from train_watermark import make_model


ATTRIBUTES = ("xyz", "opacity", "scale", "rotation", "sh")


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser("Evaluate sampled Gaussian watermarking")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modelnet_root", default="gs_data/modelsplat/modelsplat_ply")
    parser.add_argument("--dataset_file_list", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_assets", type=int, default=100)
    parser.add_argument("--sample_counts", type=parse_int_list, default=parse_int_list("512,1024,2048,4096,8192,16384"))
    parser.add_argument("--sampling_repeats", type=int, default=5)
    parser.add_argument("--base_npoints", type=int, default=4096)
    parser.add_argument("--pruning_ratios", type=parse_float_list, default=parse_float_list("0,10,30,50,70,80,90"))
    parser.add_argument("--random_pruning_repeats", type=int, default=5)
    parser.add_argument("--sample_seed", type=int, default=20260702)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    model = make_model(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def build_or_load_manifest(args, checkpoint, output_dir):
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["num_bits"] != int(checkpoint["args"]["num_bits"]):
            raise ValueError("Manifest payload length does not match checkpoint.")
        return manifest, manifest_path

    if args.dataset_file_list:
        dataset = WatermarkGaussianDataset(roots=[], npoints=args.base_npoints, file_list=args.dataset_file_list)
        selected_indices = list(range(min(args.max_assets, len(dataset))))
    else:
        dataset = WatermarkGaussianDataset(roots=[args.modelnet_root], npoints=args.base_npoints)
        split_seed = int(checkpoint["args"].get("seed", 0))
        val_ratio = float(checkpoint["args"].get("val_ratio", 0.02))
        val_len = max(1, int(len(dataset) * val_ratio))
        train_len = len(dataset) - val_len
        _, val_set = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(split_seed),
        )
        selected_indices = list(val_set.indices[: args.max_assets])
    split_seed = int(checkpoint["args"].get("seed", 0))
    val_ratio = float(checkpoint["args"].get("val_ratio", 0.02))
    message_rng = np.random.default_rng(args.sample_seed)
    num_bits = int(checkpoint["args"]["num_bits"])
    assets = []
    for dataset_index in selected_indices:
        assets.append(
            {
                "dataset_index": int(dataset_index),
                "path": dataset.files[dataset_index],
                "message_bits": message_rng.integers(0, 2, size=num_bits).astype(int).tolist(),
            }
        )
    manifest = {
        "protocol": "sampled_subset",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_bits": num_bits,
        "split_seed": split_seed,
        "sample_seed": args.sample_seed,
        "val_ratio": val_ratio,
        "num_dataset_files": len(dataset),
        "assets": assets,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, manifest_path


def sample_gaussians(gaussians, count, rng):
    replace = len(gaussians) < count
    indices = rng.choice(len(gaussians), size=count, replace=replace)
    return gaussians[indices]


def message_tensor(bits, device):
    values = torch.tensor(bits, dtype=torch.float32, device=device)
    return (values * 2.0 - 1.0).unsqueeze(0)


def accuracy(logits, message):
    correct = (logits >= 0) == (message > 0)
    return float(correct.float().mean().item()), float(correct.all(dim=1).float().mean().item())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def embed_and_extract(model, sampled, message, device):
    gaussians = torch.from_numpy(sampled).to(device=device, dtype=torch.float32).unsqueeze(0)
    synchronize(device)
    start = time.perf_counter()
    embedder_output = model.embedder(gaussians, message)
    watermarked, delta = embedder_output[:2]
    synchronize(device)
    embed_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    logits = model.extractor(watermarked)
    synchronize(device)
    extract_ms = (time.perf_counter() - start) * 1000.0
    bit_acc, word_acc = accuracy(logits, message)
    return watermarked, delta, {
        "bit_acc": bit_acc,
        "word_acc": word_acc,
        "gs_mse": float(torch.mean((watermarked - gaussians) ** 2).item()),
        "xyz_mse": float(torch.mean((watermarked[..., :3] - gaussians[..., :3]) ** 2).item()),
        "delta_abs": float(delta.abs().mean().item()),
        "embed_ms": embed_ms,
        "extract_ms": extract_ms,
    }


def extract_only(model, gaussians, message, device):
    synchronize(device)
    start = time.perf_counter()
    logits = model.extractor(gaussians)
    synchronize(device)
    extract_ms = (time.perf_counter() - start) * 1000.0
    bit_acc, word_acc = accuracy(logits, message)
    return {"bit_acc": bit_acc, "word_acc": word_acc, "extract_ms": extract_ms}


def prune_gaussians(watermarked, ratio, rule, rng):
    count = watermarked.shape[1]
    remove_count = int(round(count * ratio / 100.0))
    keep_count = max(1, count - remove_count)
    if rule == "random":
        indices = rng.permutation(count)[:keep_count]
        indices = torch.from_numpy(indices.copy()).to(watermarked.device)
    else:
        opacity = watermarked[0, :, 3]
        descending = rule == "lowest_opacity_first"
        order = torch.argsort(opacity, descending=descending)
        indices = order[:keep_count]
    return watermarked[:, indices, :]


def append_record(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def summarize(records):
    groups = defaultdict(list)
    identity_fields = ("experiment", "sample_count", "pruning_rule", "pruning_ratio")
    metric_fields = ("bit_acc", "word_acc", "gs_mse", "xyz_mse", "delta_abs", "embed_ms", "extract_ms")
    for record in records:
        key = tuple(record.get(field) for field in identity_fields)
        groups[key].append(record)

    rows = []
    for key, values in groups.items():
        row = {field: value for field, value in zip(identity_fields, key) if value is not None}
        row["num_records"] = len(values)
        for metric in metric_fields:
            samples = [float(value[metric]) for value in values if metric in value]
            if samples:
                row[f"mean_{metric}"] = float(np.mean(samples))
                row[f"std_{metric}"] = float(np.std(samples))
        rows.append(row)
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA because Gaussian-MAE grouping uses CUDA KNN.")
    device = torch.device("cuda:0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)
    manifest, manifest_path = build_or_load_manifest(args, checkpoint, output_dir)
    records_path = output_dir / "metrics_per_asset.jsonl"
    records = []

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "manifest": str(manifest_path),
                "num_assets": len(manifest["assets"]),
                "sample_counts": args.sample_counts,
                "pruning_ratios": args.pruning_ratios,
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )

    with records_path.open("w") as handle, torch.inference_mode():
        for asset_index, asset in enumerate(manifest["assets"]):
            raw = read_gaussian_ply(asset["path"], ATTRIBUTES)
            normalized = normalize_gaussian(raw, ATTRIBUTES)
            message = message_tensor(asset["message_bits"], device)

            for sample_count in args.sample_counts:
                for repeat in range(args.sampling_repeats):
                    rng = np.random.default_rng(args.sample_seed + asset_index * 100003 + sample_count * 31 + repeat)
                    sampled = sample_gaussians(normalized, sample_count, rng)
                    _, _, metrics = embed_and_extract(model, sampled, message, device)
                    record = {
                        "experiment": "sampling",
                        "asset_index": asset_index,
                        "path": asset["path"],
                        "sample_count": sample_count,
                        "repeat": repeat,
                        **metrics,
                    }
                    records.append(record)
                    append_record(handle, record)

            base_rng = np.random.default_rng(args.sample_seed + asset_index * 100003 + 17)
            base_sample = sample_gaussians(normalized, args.base_npoints, base_rng)
            watermarked, _, _ = embed_and_extract(model, base_sample, message, device)
            for rule in ("random", "lowest_opacity_first", "highest_opacity_first"):
                repeats = args.random_pruning_repeats if rule == "random" else 1
                for ratio in args.pruning_ratios:
                    for repeat in range(repeats):
                        rng = np.random.default_rng(
                            args.sample_seed + asset_index * 100003 + int(ratio * 101) + repeat
                        )
                        pruned = prune_gaussians(watermarked, ratio, rule, rng)
                        metrics = extract_only(model, pruned, message, device)
                        record = {
                            "experiment": "pruning",
                            "asset_index": asset_index,
                            "path": asset["path"],
                            "sample_count": int(pruned.shape[1]),
                            "pruning_rule": rule,
                            "pruning_ratio": ratio,
                            "repeat": repeat,
                            **metrics,
                        }
                        records.append(record)
                        append_record(handle, record)

            print(f"[asset {asset_index + 1}/{len(manifest['assets'])}] {asset['path']}", flush=True)

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
