import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset, random_split

from datasets.watermark_gaussian import WatermarkGaussianDataset


def parse_float_tuple(value):
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 5:
        raise argparse.ArgumentTypeError("Expected five budgets: xyz,opacity,scale,rotation,sh")
    return values


def parse_args():
    parser = argparse.ArgumentParser("Train a 3DGS watermark embedder/extractor")
    parser.add_argument("--modelnet_root", default="gs_data/modelsplat/modelsplat_ply")
    parser.add_argument("--shapenet_root", default="gs_data/shapesplat/shapesplat_ply")
    parser.add_argument("--objaverse_root", default="gs_data/objaverse_splats/objaverse_ply")
    parser.add_argument("--use_modelnet", action="store_true", default=True)
    parser.add_argument("--no_modelnet", action="store_false", dest="use_modelnet")
    parser.add_argument("--use_shapenet", action="store_true")
    parser.add_argument("--use_objaverse", action="store_true")
    parser.add_argument("--output_dir", default="../outputs/watermark_modelnet")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained_ckpt", default=None)
    parser.add_argument("--num_bits", type=int, default=64)
    parser.add_argument("--npoints", type=int, default=4096)
    parser.add_argument("--npoints_min", type=int, default=None)
    parser.add_argument("--npoints_max", type=int, default=None)
    parser.add_argument("--val_npoints", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lambda_msg", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=5.0)
    parser.add_argument("--lambda_xyz", type=float, default=1.0)
    parser.add_argument("--delta_scale", type=float, default=0.01)
    parser.add_argument("--embedder_type", choices=("baseline", "attribute_modulated"), default="baseline")
    parser.add_argument("--attribute_budgets", type=parse_float_tuple, default=(0.5, 1.0, 0.5, 0.25, 1.0))
    parser.add_argument("--gate_floor", type=float, default=0.25)
    parser.add_argument("--disable_carrier_gate", action="store_true")
    parser.add_argument("--noise_std", type=float, default=0.002)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--prune_max_ratio", type=float, default=0.0)
    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--gate_balance_weight", type=float, default=0.0)
    parser.add_argument("--num_group", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--trans_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dataset_file_list", default=None)
    parser.add_argument("--val_file_list", default=None)
    parser.add_argument("--balance_datasets", action="store_true")
    parser.add_argument("--samples_per_epoch", type=int, default=None)
    parser.add_argument("--modelnet_eval_file_list", default=None)
    parser.add_argument("--shapenet_eval_file_list", default=None)
    parser.add_argument("--objaverse_eval_file_list", default=None)
    parser.add_argument("--exclude_manifest", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--iter_log_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def init_distributed():
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def bit_accuracy(logits, messages):
    pred = torch.sign(logits) > 0
    target = messages > 0
    bit_acc = (pred == target).float().mean()
    word_acc = (pred == target).all(dim=1).float().mean()
    return bit_acc, word_acc


def dataset_name(path):
    if "objaverse_splats" in path:
        return "objaverse"
    if "shapesplat" in path:
        return "shapenet"
    if "modelsplat" in path:
        return "modelnet"
    raise ValueError(f"Cannot infer dataset from path: {path}")


class BalancedDistributedSampler(Sampler):
    def __init__(self, subset, num_replicas, rank, samples_per_epoch, seed=0):
        self.subset = subset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.total_size = math.ceil(samples_per_epoch / num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas
        self.groups = {}
        for subset_index, dataset_index in enumerate(subset.indices):
            path = subset.dataset.files[dataset_index]
            self.groups.setdefault(dataset_name(path), []).append(subset_index)
        if len(self.groups) < 2:
            raise ValueError("Balanced sampling requires at least two recognized datasets.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        group_count = len(self.groups)
        per_group = math.ceil(self.total_size / group_count)
        indices = []
        for name in sorted(self.groups):
            group = self.groups[name]
            if len(group) >= per_group:
                selected = torch.randperm(len(group), generator=generator)[:per_group].tolist()
            else:
                selected = torch.randint(len(group), (per_group,), generator=generator).tolist()
            indices.extend(group[index] for index in selected)
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[index] for index in permutation[: self.total_size]]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples


def build_loaders(args, distributed):
    roots = []
    if args.use_modelnet:
        roots.append(args.modelnet_root)
    if args.use_shapenet:
        roots.append(args.shapenet_root)
    if args.use_objaverse:
        roots.append(args.objaverse_root)

    dataset_npoints = args.npoints_max if args.npoints_max is not None else args.npoints
    if args.npoints_min is not None and args.npoints_min > dataset_npoints:
        raise ValueError("npoints_min cannot exceed npoints_max/npoints.")
    if args.val_npoints is not None and args.val_npoints > dataset_npoints:
        raise ValueError("val_npoints cannot exceed npoints_max/npoints.")

    if args.val_file_list:
        if not args.dataset_file_list:
            raise ValueError("--val_file_list requires --dataset_file_list")
        if args.exclude_manifest:
            raise ValueError("Apply exclusions before materializing explicit train/val lists.")
        train_dataset = WatermarkGaussianDataset(
            roots=[], npoints=dataset_npoints, file_list=args.dataset_file_list
        )
        val_dataset = WatermarkGaussianDataset(
            roots=[], npoints=dataset_npoints, file_list=args.val_file_list
        )
        overlap = set(train_dataset.files) & set(val_dataset.files)
        if overlap:
            raise ValueError(f"Explicit train/val lists overlap by {len(overlap)} paths")
        train_set = Subset(train_dataset, range(len(train_dataset)))
        val_set = Subset(val_dataset, range(len(val_dataset)))
        dataset_size = len(train_dataset) + len(val_dataset)
    else:
        dataset = WatermarkGaussianDataset(
            roots=roots,
            npoints=dataset_npoints,
            max_files=args.max_files,
            file_list=args.dataset_file_list,
        )
        if args.exclude_manifest:
            manifest = json.loads(Path(args.exclude_manifest).read_text())
            excluded = {asset["source_path"] for asset in manifest.get("assets", [])}
            before = len(dataset.files)
            dataset.files = [path for path in dataset.files if path not in excluded]
            if is_main_process():
                print(
                    f"Excluded {before - len(dataset.files)} evaluation assets from {args.exclude_manifest}",
                    flush=True,
                )
        val_len = max(1, int(len(dataset) * args.val_ratio))
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed),
        )
        dataset_size = len(dataset)
    if args.balance_datasets:
        replicas = dist.get_world_size() if distributed else 1
        sampler_rank = dist.get_rank() if distributed else 0
        samples_per_epoch = args.samples_per_epoch or len(train_set)
        train_sampler = BalancedDistributedSampler(
            train_set,
            num_replicas=replicas,
            rank=sampler_rank,
            samples_per_epoch=samples_per_epoch,
            seed=args.seed,
        )
    else:
        train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader, dataset_size, train_sampler


def build_eval_loaders(args, distributed):
    eval_lists = {
        "modelnet": args.modelnet_eval_file_list,
        "shapenet": args.shapenet_eval_file_list,
        "objaverse": args.objaverse_eval_file_list,
    }
    loaders = {}
    eval_npoints = args.val_npoints or args.npoints
    eval_workers = max(1, args.workers // 2) if args.workers > 0 else 0
    for offset, (name, file_list) in enumerate(eval_lists.items()):
        if not file_list:
            continue
        dataset = WatermarkGaussianDataset(
            roots=[],
            npoints=eval_npoints,
            file_list=file_list,
            deterministic=True,
            seed=args.seed + 10000 * (offset + 1),
        )
        sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=eval_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if eval_workers > 0 else None,
            drop_last=False,
        )
    return loaders


def make_model(args):
    from models.Gaussian_Watermark import GaussianWatermarkModel

    return GaussianWatermarkModel(
        num_bits=args.num_bits,
        delta_scale=args.delta_scale,
        num_group=args.num_group,
        group_size=args.group_size,
        trans_dim=args.trans_dim,
        encoder_dims=args.trans_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        embedder_type=getattr(args, "embedder_type", "baseline"),
        attribute_budgets=getattr(args, "attribute_budgets", (0.5, 1.0, 0.5, 0.25, 1.0)),
        gate_floor=getattr(args, "gate_floor", 0.25),
        disable_carrier_gate=getattr(args, "disable_carrier_gate", False),
    )


def load_pretrained_backbone(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("base_model", ckpt)
    cleaned = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        if key.startswith("base_model."):
            key = key[len("base_model.") :]
        if key.startswith("MAE_encoder."):
            key = key[len("MAE_encoder.") :]
        if key.startswith(("cls_token", "cls_pos", "encoder.", "pos_embed.", "blocks.", "norm.")):
            cleaned[key] = value

    missing = sorted(set(model.extractor.backbone.state_dict().keys()) - set(cleaned.keys()))
    unexpected = sorted(set(cleaned.keys()) - set(model.extractor.backbone.state_dict().keys()))
    if missing or unexpected:
        raise RuntimeError(
            "Pretrained backbone keys do not match strict=True load.\n"
            f"Missing: {missing[:20]}\n"
            f"Unexpected: {unexpected[:20]}"
        )
    model.extractor.backbone.load_state_dict(cleaned, strict=True)
    if is_main_process():
        print(f"Loaded pretrained Gaussian-MAE backbone with strict=True: {ckpt_path}", flush=True)


def reduce_metrics(metrics, batch, device):
    if not dist.is_available() or not dist.is_initialized():
        return {key: float(value.item()) * batch for key, value in metrics.items()}, batch

    keys = list(metrics.keys())
    values = torch.tensor([float(metrics[key].item()) * batch for key in keys] + [float(batch)], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    reduced_batch = int(values[-1].item())
    return {key: float(values[i].item()) for i, key in enumerate(keys)}, reduced_batch


def fixed_messages(paths, num_bits, seed, device):
    rows = []
    for path in paths:
        digest = hashlib.shake_256(f"{seed}:{path}".encode()).digest((num_bits + 7) // 8)
        bits = [(digest[index // 8] >> (index % 8)) & 1 for index in range(num_bits)]
        rows.append(bits)
    return torch.tensor(rows, dtype=torch.float32, device=device) * 2.0 - 1.0


def run_epoch(model, loader, optimizer, args, device, train, epoch, phase=None, fixed_eval_messages=False):
    model.train(train)
    totals = {}
    count = 0
    phase = phase or ("train" if train else "val")
    start_time = time.time()
    end_time = start_time
    iteration_rows = []

    for it, (paths, gaussians) in enumerate(loader):
        iter_start = time.time()
        data_sec = iter_start - end_time
        gaussians = gaussians.to(device, non_blocking=True)
        min_npoints = args.npoints_min if args.npoints_min is not None else args.npoints
        max_npoints = args.npoints_max if args.npoints_max is not None else args.npoints
        if train and max_npoints > min_npoints:
            sampled_npoints = torch.randint(min_npoints, max_npoints + 1, (1,), device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(sampled_npoints, src=0)
            active_npoints = int(sampled_npoints.item())
        elif not train and args.val_npoints is not None:
            active_npoints = args.val_npoints
        else:
            active_npoints = max_npoints
        if active_npoints < gaussians.shape[1]:
            point_indices = torch.randperm(gaussians.shape[1], device=device)[:active_npoints]
            gaussians = gaussians[:, point_indices]
        if fixed_eval_messages:
            messages = fixed_messages(paths, args.num_bits, args.seed, device)
        else:
            messages = (torch.randint(0, 2, (gaussians.shape[0], args.num_bits), device=device).float() * 2.0) - 1.0
        prune_ratio = 0.0
        if train and args.prune_max_ratio > 0:
            prune_ratio = float(torch.rand((), device=device).item()) * args.prune_max_ratio

        with torch.set_grad_enabled(train):
            logits, watermarked, _, delta = model(
                gaussians,
                messages,
                noise_std=args.noise_std if train else 0.0,
                dropout_prob=args.dropout_prob if train else 0.0,
                prune_ratio=prune_ratio,
            )
            loss_msg = F.binary_cross_entropy_with_logits(logits, (messages + 1.0) * 0.5)
            loss_consistency = logits.new_zeros(())
            if train and args.consistency_weight > 0:
                logits_view2, _, _, _ = model(
                    gaussians,
                    messages,
                    noise_std=args.noise_std,
                    dropout_prob=args.dropout_prob,
                    prune_ratio=prune_ratio,
                )
                loss_msg_view2 = F.binary_cross_entropy_with_logits(
                    logits_view2,
                    (messages + 1.0) * 0.5,
                )
                loss_msg = 0.5 * (loss_msg + loss_msg_view2)
                loss_consistency = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(logits_view2))

            model_without_wrapper = model.module if hasattr(model, "module") else model
            carrier_gate = getattr(model_without_wrapper, "last_carrier_gate", None)
            loss_gate_balance = logits.new_zeros(())
            if carrier_gate is not None:
                loss_gate_balance = torch.mean((carrier_gate - 1.0) ** 2)
            loss_delta = F.mse_loss(watermarked, gaussians)
            loss_xyz = F.mse_loss(watermarked[..., :3], gaussians[..., :3])
            loss = (
                args.lambda_msg * loss_msg
                + args.lambda_delta * loss_delta
                + args.lambda_xyz * loss_xyz
                + args.consistency_weight * loss_consistency
                + args.gate_balance_weight * loss_gate_balance
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bit_acc, word_acc = bit_accuracy(logits.detach(), messages)
        batch = gaussians.shape[0]
        metrics = {
            "loss": loss.detach(),
            "message_loss": loss_msg.detach(),
            "gs_loss": loss_delta.detach(),
            "xyz_loss": loss_xyz.detach(),
            "consistency_loss": loss_consistency.detach(),
            "gate_balance_loss": loss_gate_balance.detach(),
            "bit_acc": bit_acc.detach(),
            "word_acc": word_acc.detach(),
            "delta_abs": delta.detach().abs().mean(),
            "prune_ratio": torch.tensor(prune_ratio, device=device),
            "npoints": torch.tensor(float(active_npoints), device=device),
        }
        reduced, reduced_batch = reduce_metrics(metrics, batch, device)
        for key, value in reduced.items():
            totals[key] = totals.get(key, 0.0) + value
        count += reduced_batch

        iter_compute_sec = time.time() - iter_start
        if (
            is_main_process()
            and args.iter_log_freq > 0
            and (it % args.iter_log_freq == 0 or it + 1 == len(loader))
        ):
            iteration_rows.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "iter": it + 1,
                    "num_iters": len(loader),
                    "global_step": epoch * len(loader) + it + 1 if train else None,
                    "epoch_progress": epoch + (it + 1) / len(loader),
                    "data_sec": data_sec,
                    "compute_sec": iter_compute_sec,
                    "epoch_elapsed_sec": time.time() - start_time,
                    **{
                        key: value / max(reduced_batch, 1)
                        for key, value in reduced.items()
                    },
                }
            )

        if is_main_process() and args.log_freq > 0 and (it % args.log_freq == 0 or it + 1 == len(loader)):
            elapsed = time.time() - start_time
            print(
                (
                    f"[{phase} epoch {epoch}/{args.epochs}] [iter {it + 1}/{len(loader)}] "
                    f"loss {metrics['loss'].item():.6f}, "
                    f"message_loss {metrics['message_loss'].item():.6f}, "
                    f"gs_loss {metrics['gs_loss'].item():.6f}, "
                    f"xyz_loss {metrics['xyz_loss'].item():.6f}, "
                    f"consistency_loss {metrics['consistency_loss'].item():.6f}, "
                    f"gate_balance_loss {metrics['gate_balance_loss'].item():.6f}, "
                    f"prune_ratio {prune_ratio:.4f}, "
                    f"npoints {active_npoints}, "
                    f"bit_acc {metrics['bit_acc'].item():.4f}, "
                    f"word_acc {metrics['word_acc'].item():.4f}, "
                    f"delta_abs {metrics['delta_abs'].item():.6f}, "
                    f"data_sec {data_sec:.2f}, "
                    f"compute_sec {time.time() - iter_start:.2f}, "
                    f"elapsed_sec {elapsed:.2f}"
                ),
                flush=True,
            )
        end_time = time.time()

    if iteration_rows:
        iter_log_path = Path(args.output_dir) / "iter_log.jsonl"
        with iter_log_path.open("a") as handle:
            for row in iteration_rows:
                handle.write(json.dumps(row) + "\n")

    return {key: value / max(count, 1) for key, value in totals.items()}


def prepare_resume_logs(log_path, iter_log_path, start_epoch):
    completed_time_sec = 0.0
    if log_path.is_file():
        seen_epochs = set()
        with log_path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    epoch = int(row["epoch"])
                    if epoch < start_epoch and epoch not in seen_epochs:
                        completed_time_sec += float(row.get("time_sec", 0.0))
                        seen_epochs.add(epoch)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    if iter_log_path.is_file():
        retained = []
        with iter_log_path.open() as handle:
            for line in handle:
                try:
                    if int(json.loads(line)["epoch"]) < start_epoch:
                        retained.append(line)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        with iter_log_path.open("w") as handle:
            handle.writelines(retained)
    return completed_time_sec


def save_checkpoint(path, model, optimizer, epoch, args, best=False):
    if not is_main_process():
        return
    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }

    def atomic_save(destination):
        temporary = f"{destination}.tmp.{os.getpid()}"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    atomic_save(path)
    if best:
        atomic_save(os.path.join(os.path.dirname(path), "checkpoint_best.pth"))


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gaussian watermark training requires CUDA because Gaussian_MAE grouping uses knn_cuda.")
    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_files, train_sampler = build_loaders(args, distributed)
    eval_loaders = build_eval_loaders(args, distributed)
    model = make_model(args).to(device)
    if args.pretrained_ckpt:
        load_pretrained_backbone(model, args.pretrained_ckpt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_bit_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1

    if distributed:
        # Consistency training runs two forwards before one backward. DDP's
        # per-forward BN-buffer broadcast would invalidate the first graph.
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        if is_main_process():
            print(f"Using DistributedDataParallel with world_size={world_size}", flush=True)
    elif torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = torch.nn.DataParallel(model)

    if is_main_process():
        print(
            json.dumps(
                {
                    "args": vars(args),
                    "num_files": num_files,
                    "eval_files": {name: len(loader.dataset) for name, loader in eval_loaders.items()},
                    "device": str(device),
                    "distributed": distributed,
                    "world_size": world_size,
                },
                indent=2,
            ),
            flush=True,
        )
    log_path = Path(args.output_dir) / "log.jsonl"
    iter_log_path = Path(args.output_dir) / "iter_log.jsonl"
    completed_time_sec = prepare_resume_logs(log_path, iter_log_path, start_epoch)
    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, optimizer, args, device, train=True, epoch=epoch)
        should_validate = args.val_freq > 0 and (epoch % args.val_freq == 0 or epoch + 1 == args.epochs)
        if should_validate:
            with torch.no_grad():
                val_stats = run_epoch(model, val_loader, optimizer, args, device, train=False, epoch=epoch)
                eval_stats = {
                    name: run_epoch(
                        model,
                        loader,
                        optimizer,
                        args,
                        device,
                        train=False,
                        epoch=epoch,
                        phase=f"test-{name}",
                        fixed_eval_messages=True,
                    )
                    for name, loader in eval_loaders.items()
                }
        else:
            val_stats = {}
            eval_stats = {}

        is_best = bool(val_stats) and val_stats["bit_acc"] > best_bit_acc
        if val_stats:
            best_bit_acc = max(best_bit_acc, val_stats["bit_acc"])
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - t0, 2),
            "cumulative_time_sec": round(completed_time_sec + time.time() - training_start_time, 2),
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            **{
                f"test_{dataset}_{metric}": value
                for dataset, stats in eval_stats.items()
                for metric, value in stats.items()
            },
            "best_bit_acc": best_bit_acc,
        }
        if is_main_process():
            if should_validate:
                print(json.dumps(row, sort_keys=True), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        save_checkpoint(os.path.join(args.output_dir, "checkpoint_latest.pth"), model, optimizer, epoch, args, best=is_best)
        if args.save_freq and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pth"), model, optimizer, epoch, args)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
