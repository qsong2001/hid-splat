# Exact Split for the HiD-Splat Mixed Run

This directory preserves the exact dataset membership used by
`outputs/watermark_mixed_v2_dynamic_4k12k`.

- The 205,226-entry candidate pool is canonicalized with
  `sorted(set(paths))`, matching `WatermarkGaussianDataset`.
- The exact train/validation membership is materialized from PyTorch
  `random_split` with `seed=0` and `val_ratio=0.02`.
- The domain test lists are preserved at their original sizes: 100 ModelNet,
  520 ShapeNet, and 100 Objaverse assets.
- Train, validation, and test membership is disjoint within each dataset.
- All paths are relative to the `ShapeSplat-Gaussian_MAE` repository root.
- `metadata.json` records counts and SHA-256 hashes for verification.

Use the exact split without recomputing membership:

```bash
--dataset_file_list splits/hidsplat_mixed_run_v1/train.txt \
--val_file_list splits/hidsplat_mixed_run_v1/val.txt \
--modelnet_eval_file_list splits/hidsplat_mixed_run_v1/modelnet_test.txt \
--shapenet_eval_file_list splits/hidsplat_mixed_run_v1/shapenet_test.txt \
--objaverse_eval_file_list splits/hidsplat_mixed_run_v1/objaverse_test.txt
```

Regenerate the files from the original run artifacts with:

```bash
python scripts/materialize_release_splits.py \
  --source-dir outputs/watermark_mixed_v2_dynamic_4k12k/splits \
  --output-dir ShapeSplat-Gaussian_MAE/splits/hidsplat_mixed_run_v1 \
  --name hidsplat_mixed_run_v1 \
  --all-test-candidates
```
