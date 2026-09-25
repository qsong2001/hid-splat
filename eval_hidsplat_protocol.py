import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets.watermark_gaussian import normalize_gaussian, read_gaussian_ply
from eval_watermark_sampling import load_model


ATTRIBUTES = ("xyz", "opacity", "scale", "rotation", "sh")


def parse_args():
    parser = argparse.ArgumentParser("Unified HiD-Splat component, robustness, and detection evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("clean", "component", "robustness", "detection"), required=True)
    parser.add_argument("--point_count", type=int, default=4096)
    parser.add_argument("--max_assets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--random_pruning_repeats", type=int, default=5)
    parser.add_argument("--pruning_ratios", default="10,20,30,40,50")
    parser.add_argument("--pruning_only", action="store_true")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def parse_ratios(value):
    ratios = tuple(int(item) for item in value.split(",") if item.strip())
    if not ratios or any(ratio <= 0 or ratio >= 100 for ratio in ratios):
        raise ValueError(f"Invalid pruning ratios: {value}")
    return ratios


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def message_for_asset(asset, num_bits, seed, asset_index):
    messages = asset.get("messages", [])
    if messages and len(messages[0]) == num_bits:
        bits = np.asarray(messages[0], dtype=np.float32)
    else:
        rng = np.random.default_rng(seed + asset_index * 100003 + num_bits * 1009)
        bits = rng.integers(0, 2, size=num_bits).astype(np.float32)
    return bits


def points_for_asset(asset, cache_dir, point_count, seed, asset_index):
    cached = np.load(cache_dir / asset["cache_path"], mmap_mode="r")
    if point_count <= cached.shape[1]:
        return np.asarray(cached[0, :point_count], dtype=np.float32)
    raw = read_gaussian_ply(asset["source_path"], ATTRIBUTES)
    normalized = normalize_gaussian(raw, ATTRIBUTES)
    rng = np.random.default_rng(seed + asset_index * 100003 + point_count * 31)
    indices = rng.choice(len(normalized), size=point_count, replace=len(normalized) < point_count)
    return np.asarray(normalized[indices], dtype=np.float32)


def message_tensor(bits, device):
    tensor = torch.from_numpy(bits).to(device=device, dtype=torch.float32)
    return (tensor * 2.0 - 1.0).unsqueeze(0)


def decode(model, gaussians):
    return model.extractor(gaussians.contiguous())


def decode_metrics(logits, message):
    correct = (logits >= 0) == (message > 0)
    return {
        "bit_acc": float(correct.float().mean().item()),
        "word_acc": float(correct.all(dim=1).float().mean().item()),
        "match_count": int(correct[0].sum().item()),
    }


def prune(points, ratio, rule, rng):
    count = points.shape[1]
    keep_count = max(1, count - int(round(count * ratio / 100.0)))
    if rule == "random":
        indices = torch.from_numpy(rng.permutation(count)[:keep_count].copy()).to(points.device)
    else:
        opacity = points[0, :, 3]
        keep_highest = rule == "lowest_opacity_first"
        indices = torch.argsort(opacity, descending=keep_highest)[:keep_count]
    return points[:, indices, :]


def quantize(points, bits):
    levels = float((1 << bits) - 1)
    low = points.amin(dim=1, keepdim=True)
    high = points.amax(dim=1, keepdim=True)
    span = torch.clamp(high - low, min=1e-6)
    result = torch.round((points - low) / span * levels) / levels * span + low
    rotations = result[..., 7:11]
    rotations = rotations / torch.clamp(torch.linalg.norm(rotations, dim=-1, keepdim=True), min=1e-9)
    signs = torch.sign(rotations[..., :1])
    signs[signs == 0] = 1
    result[..., 7:11] = rotations * signs
    return result


def spatial_crop(points, retain_percent):
    keep_count = max(1, int(round(points.shape[1] * retain_percent / 100.0)))
    projection = points[0, :, 0] + 0.37 * points[0, :, 1] - 0.19 * points[0, :, 2]
    indices = torch.argsort(projection)[:keep_count]
    return points[:, indices, :]


def similarity_transform(points, angle_degrees=30.0, scale=1.1):
    result = points.clone()
    angle = math.radians(angle_degrees)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        device=points.device,
        dtype=points.dtype,
    )
    result[..., :3] = result[..., :3] @ rotation.T * scale
    return result


def condition_sets(
    mode,
    watermarked,
    seed,
    asset_index,
    random_repeats,
    pruning_ratios=None,
    pruning_only=False,
):
    yield "clean", watermarked, 0
    if mode == "clean":
        return
    if mode == "component":
        for repeat in range(random_repeats):
            rng = np.random.default_rng(seed + asset_index * 100003 + repeat)
            yield "prune_random_50", prune(watermarked, 50, "random", rng), repeat
        for rule in ("lowest_opacity_first", "highest_opacity_first"):
            rng = np.random.default_rng(seed + asset_index * 100003)
            yield f"prune_{rule}_50", prune(watermarked, 50, rule, rng), 0
        return
    if mode in ("robustness", "detection"):
        ratios = pruning_ratios if mode == "robustness" else (30,)
        rules = ("random", "lowest_opacity_first", "highest_opacity_first") if mode == "robustness" else ("random",)
        for rule in rules:
            for ratio in ratios:
                repeats = random_repeats if rule == "random" else 1
                for repeat in range(repeats):
                    rng = np.random.default_rng(seed + asset_index * 100003 + ratio * 101 + repeat)
                    yield f"prune_{rule}_{ratio}", prune(watermarked, ratio, rule, rng), repeat
        if pruning_only:
            return
        quant_bits = (16, 12, 8) if mode == "robustness" else (8,)
        for bits in quant_bits:
            yield f"quantize_{bits}bit", quantize(watermarked, bits), 0
    if mode != "robustness":
        return
    for sigma in (0.001, 0.002, 0.005):
        generator = torch.Generator(device=watermarked.device).manual_seed(seed + asset_index * 100003 + int(sigma * 1e6))
        noisy = watermarked.clone()
        noisy[..., :3] += torch.randn(noisy[..., :3].shape, generator=generator, device=noisy.device) * sigma
        yield f"xyz_noise_{sigma:g}", noisy, 0
    for sigma in (0.001, 0.002, 0.005):
        generator = torch.Generator(device=watermarked.device).manual_seed(seed + asset_index * 100003 + int(sigma * 2e6))
        noisy = watermarked + torch.randn(watermarked.shape, generator=generator, device=watermarked.device) * sigma
        yield f"attribute_noise_{sigma:g}", noisy, 0
    for retain in (90, 70, 50):
        yield f"spatial_crop_retain_{retain}", spatial_crop(watermarked, retain), 0
    rng = np.random.default_rng(seed + asset_index * 100003 + 991)
    indices = torch.from_numpy(rng.permutation(watermarked.shape[1]).copy()).to(watermarked.device)
    yield "permutation", watermarked[:, indices, :], 0
    yield "similarity_rot30_scale1.1", similarity_transform(watermarked), 0


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["condition"])].append(record)
    rows = []
    metrics = ("bit_acc", "word_acc", "normalized_gs_shift", "normalized_all_attribute_shift", "gs_mse", "xyz_mse")
    for (dataset, condition), values in sorted(groups.items()):
        row = {"dataset": dataset, "condition": condition, "num_records": len(values)}
        for metric in metrics:
            samples = np.asarray([value[metric] for value in values if metric in value], dtype=np.float64)
            if len(samples):
                row[f"mean_{metric}"] = float(samples.mean())
                row[f"std_{metric}"] = float(samples.std())
        rows.append(row)
    return rows


def binomial_threshold(num_bits, target_fpr):
    denominator = float(2**num_bits)
    for matches in range(num_bits + 1):
        tail = sum(math.comb(num_bits, k) for k in range(matches, num_bits + 1)) / denominator
        if tail <= target_fpr:
            return matches, tail
    return num_bits + 1, 0.0


def detection_summary(records, num_bits):
    positives = defaultdict(list)
    negatives = defaultdict(list)
    for record in records:
        group = positives if record["label"] == "positive" else negatives
        group[record["condition"]].append(record["match_count"])
    thresholds = []
    for fpr in (1e-2, 1e-3, 1e-4):
        matches, theoretical_fpr = binomial_threshold(num_bits, fpr)
        row = {"target_fpr": fpr, "match_threshold": matches, "theoretical_fpr": theoretical_fpr}
        for condition, scores in positives.items():
            row[f"tpr_{condition}"] = float(np.mean(np.asarray(scores) >= matches))
        for condition, scores in negatives.items():
            row[f"empirical_fpr_{condition}"] = float(np.mean(np.asarray(scores) >= matches))
        thresholds.append(row)
    return {
        "num_bits": num_bits,
        "thresholds": thresholds,
        "histograms": {
            "positive": {key: np.bincount(values, minlength=num_bits + 1).tolist() for key, values in positives.items()},
            "negative": {key: np.bincount(values, minlength=num_bits + 1).tolist() for key, values in negatives.items()},
        },
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by Gaussian-MAE grouping.")
    device = torch.device("cuda:0")
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"Invalid shard {args.shard_id}/{args.num_shards}")
    all_assets = manifest["assets"][: args.max_assets or None]
    indexed_assets = [
        (asset_index, asset)
        for asset_index, asset in enumerate(all_assets)
        if asset_index % args.num_shards == args.shard_id
    ]
    pruning_ratios = parse_ratios(args.pruning_ratios)
    model, checkpoint = load_model(args.checkpoint, device)
    num_bits = int(checkpoint["args"]["num_bits"])
    records = []
    records_path = output_dir / "metrics_per_asset.jsonl"

    with records_path.open("w") as handle, torch.inference_mode():
        for local_index, (asset_index, asset) in enumerate(indexed_assets):
            points = points_for_asset(asset, cache_dir, args.point_count, args.seed, asset_index)
            base = torch.from_numpy(points).to(device=device, dtype=torch.float32).unsqueeze(0)
            bits = message_for_asset(asset, num_bits, args.seed, asset_index)
            message = message_tensor(bits, device)
            watermarked, delta = model.embedder(base, message)[:2]
            intrinsic = {
                "normalized_gs_shift": float(torch.linalg.norm(watermarked[..., :3] - base[..., :3], dim=-1).mean().item()),
                "normalized_all_attribute_shift": float(torch.linalg.norm(watermarked - base, dim=-1).mean().item()),
                "gs_mse": float(torch.mean((watermarked - base) ** 2).item()),
                "xyz_mse": float(torch.mean((watermarked[..., :3] - base[..., :3]) ** 2).item()),
                "delta_abs": float(delta.abs().mean().item()),
            }
            if args.mode == "detection":
                negative_specs = (("unwatermarked", base),)
                for condition, attacked, repeat in condition_sets("detection", watermarked, args.seed, asset_index, args.random_pruning_repeats):
                    metrics = decode_metrics(decode(model, attacked), message)
                    record = {"dataset": asset["dataset"], "asset_index": asset_index, "condition": condition, "repeat": repeat, "label": "positive", **metrics}
                    records.append(record)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                for condition, negative in negative_specs:
                    metrics = decode_metrics(decode(model, negative), message)
                    record = {"dataset": asset["dataset"], "asset_index": asset_index, "condition": condition, "repeat": 0, "label": "negative", **metrics}
                    records.append(record)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                wrong_rng = np.random.default_rng(args.seed + asset_index * 100003 + num_bits * 9176 + 41)
                wrong_bits = wrong_rng.integers(0, 2, size=num_bits).astype(np.float32)
                if np.array_equal(wrong_bits, bits):
                    wrong_bits[0] = 1.0 - wrong_bits[0]
                wrong_message = message_tensor(wrong_bits, device)
                metrics = decode_metrics(decode(model, watermarked), wrong_message)
                record = {"dataset": asset["dataset"], "asset_index": asset_index, "condition": "wrong_owner_message", "repeat": 0, "label": "negative", **metrics}
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            else:
                for condition, attacked, repeat in condition_sets(
                    args.mode,
                    watermarked,
                    args.seed,
                    asset_index,
                    args.random_pruning_repeats,
                    pruning_ratios=pruning_ratios,
                    pruning_only=args.pruning_only,
                ):
                    metrics = decode_metrics(decode(model, attacked), message)
                    record = {
                        "dataset": asset["dataset"],
                        "asset_index": asset_index,
                        "source_path": asset["source_path"],
                        "condition": condition,
                        "repeat": repeat,
                        "point_count": int(attacked.shape[1]),
                        **intrinsic,
                        **metrics,
                    }
                    records.append(record)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[asset {local_index + 1}/{len(indexed_assets)}; global={asset_index}] "
                f"{asset['dataset']} {asset['source_path']}",
                flush=True,
            )

    summary = {
        "args": vars(args),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_assets": len(indexed_assets),
        "num_assets_total": len(all_assets),
        "num_bits": num_bits,
    }
    if args.mode == "detection":
        summary.update(detection_summary(records, num_bits))
    else:
        summary["results"] = summarize(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
