# HiD-Splat: One-Pass Attribute-Space Watermarking for 3D Gaussian Splatting

This directory contains the core code for the message-conditioned Gaussian embedder and Gaussian-MAE-based extractor, mixed-domain training, and Gaussian-space evaluation. It is a source release bundle, not a copy of the datasets, pretrained weights, trained checkpoints, or per-scene baseline repositories.

## Contents

| Path | Purpose |
| --- | --- |
| `models/Gaussian_Watermark.py` | Attribute-modulated embedder, carrier gate, and extractor |
| `models/transformer.py`, `models/neural_nn.py`, `models/ops.py` | Gaussian-MAE grouping and transformer backbone |
| `datasets/watermark_gaussian.py` | PLY reading and Gaussian normalization |
| `train_watermark.py` | Single-GPU or PyTorch DDP training, validation, and checkpointing |
| `prepare_watermark_eval_cache.py` | Fixed-message/subset evaluation cache |
| `eval_watermark_sampling.py` | Cardinality and pruning evaluation |
| `eval_hidsplat_protocol.py` | Clean, component, robustness, and detection evaluation |
| `eval_watermark_cross_dataset.py` | Cross-dataset evaluation |
| `splits/` | Recorded train/validation/test file lists and metadata |

## Environment

Use Linux, an NVIDIA GPU, a matching CUDA toolkit, and Python 3.10. The experiments used PyTorch 2.2.0 with CUDA 12.1, NumPy 1.25.0, `plyfile` 1.1.3, and `timm` 1.0.27. Install PyTorch for your CUDA driver first, then:

```bash
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/unique1i/pointnet2_pytorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib'
python -m pip install 'git+https://github.com/unique1i/knn_cuda.git#egg=KNN_CUDA'
```

The last two packages compile CUDA extensions. Their source URLs follow the Gaussian-MAE environment. Build them against the *same* PyTorch/CUDA environment used for training. A CUDA GPU is required for model training and evaluation.

## Data and splits

Run commands from this directory. The split files store paths relative to the working directory, for example `gs_data/modelsplat/modelsplat_ply/...`. Place or symlink `gs_data/` here so that it contains:

```text
gs_data/modelsplat/modelsplat_ply/
gs_data/shapesplat/shapesplat_ply/
gs_data/objaverse_splats/objaverse_ply/
```

`splits/hidsplat_v1/` provides train/validation/test lists with 100 test assets in each domain. `splits/hidsplat_mixed_run_v1/` records the original mixed run exactly; its ShapeNet test list has 520 assets, while ModelNet and Objaverse each have 100. The train and validation lists are identical between these split versions. Their `metadata.json` files record counts and hashes. Use `hidsplat_v1` for the balanced three-domain test protocol. Data and checkpoints must be obtained separately; neither is redistributed here.

## Pretraining and mixed-domain training

Place the official Gaussian-MAE ModelNet checkpoint under `pretrained/gaussian_mae_ckpts/` with its original filename:

```text
release_finetune_modelnet40_enc_full_4k_pretrain_1k_best_acc93.43.pth
```

The extractor backbone is loaded with `strict=True`; a missing or incompatible checkpoint is an error. The following is the mixed-domain 48-bit configuration. `--batch_size` is per GPU under DDP.

```bash
torchrun --standalone --nproc_per_node=8 train_watermark.py \
  --use_shapenet --use_objaverse \
  --dataset_file_list splits/hidsplat_v1/train.txt \
  --val_file_list splits/hidsplat_v1/val.txt \
  --modelnet_eval_file_list splits/hidsplat_v1/modelnet_test.txt \
  --shapenet_eval_file_list splits/hidsplat_v1/shapenet_test.txt \
  --objaverse_eval_file_list splits/hidsplat_v1/objaverse_test.txt \
  --pretrained_ckpt pretrained/gaussian_mae_ckpts/release_finetune_modelnet40_enc_full_4k_pretrain_1k_best_acc93.43.pth \
  --output_dir outputs/mixed_48bit \
  --embedder_type attribute_modulated --num_bits 48 \
  --npoints 12288 --npoints_min 4096 --npoints_max 12288 --val_npoints 4096 \
  --attribute_budgets 0.5,1.0,0.5,0.25,1.0 --gate_floor 0.25 \
  --prune_max_ratio 0.5 --consistency_weight 1.0 --gate_balance_weight 0.01 \
  --balance_datasets --samples_per_epoch 36000 \
  --batch_size 64 --workers 8 --epochs 170 --val_freq 5 --save_freq 5
```

For fewer GPUs, change `--nproc_per_node` and, if needed, per-GPU batch size and worker count. Training logs are written to `log.jsonl` and `iter_log.jsonl`; `--resume outputs/mixed_48bit/checkpoint_latest.pth` resumes a checkpoint. The command above is a reproducible configuration, not a claim that retraining with another world size or split reproduces an existing checkpoint bit-for-bit.

## Gaussian-space evaluation

The cache fixes test assets, sampled subsets, and messages. Then run the desired evaluation mode with the same checkpoint:

```bash
python prepare_watermark_eval_cache.py \
  --modelnet_file_list splits/hidsplat_v1/modelnet_test.txt \
  --shapenet_file_list splits/hidsplat_v1/shapenet_test.txt \
  --objaverse_file_list splits/hidsplat_v1/objaverse_test.txt \
  --output_dir eval_cache

python eval_hidsplat_protocol.py \
  --checkpoint outputs/mixed_48bit/checkpoint_best.pth \
  --cache_dir eval_cache --output_dir outputs/eval_clean \
  --mode clean --point_count 8192

python eval_watermark_sampling.py \
  --checkpoint outputs/mixed_48bit/checkpoint_best.pth \
  --dataset_file_list splits/hidsplat_v1/modelnet_test.txt \
  --output_dir outputs/eval_sampling \
  --sample_counts 4096,8192,12288,16384,24576,36864,49152 \
  --pruning_ratios 0,10,20,30,40,50
```

See `--help` for `component`, `robustness`, and `detection` modes and sharding options. These scripts evaluate attribute-space bit recovery; they do **not** calculate rendered PSNR/SSIM/LPIPS. Matched-view rendering and the four per-scene baseline integrations depend on separate renderer/baseline projects and are not part of this core bundle. Do not compare these Gaussian-space outputs as if they were rendered fidelity measurements.

## Attribution

The Gaussian-MAE backbone and associated model utilities are adapted from ShapeSplats / Gaussian-MAE. The source license is retained in `LICENSE-ShapeSplats`. Cite the HiD-Splat paper and the Gaussian-MAE/ShapeSplats work when using this code. Before public release, review third-party CUDA-extension and dataset licenses independently.
