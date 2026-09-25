# HiD-Splat v1 Splits

These portable file lists materialize the exact train/validation split used by
the mixed-domain HiD-Splat run. Paths are relative to the
`ShapeSplat-Gaussian_MAE` repository root.

- The candidate training pool was canonicalized with `sorted(set(paths))`.
- Training and validation were split globally with PyTorch `random_split`,
  `seed=0`, and `val_ratio=0.02`, matching `train_watermark.py`.
- Each dataset has exactly 100 fixed test scenes selected by
  `outputs/baseline_packages/manifest.json`, so HiD-Splat and every baseline
  use identical assets.
- Train, validation, and test paths are disjoint within every dataset.
- `metadata.json` records counts and SHA-256 hashes for release verification.

To train without recomputing the split, pass:

```bash
--dataset_file_list splits/hidsplat_v1/train.txt \
--val_file_list splits/hidsplat_v1/val.txt \
--modelnet_eval_file_list splits/hidsplat_v1/modelnet_test.txt \
--shapenet_eval_file_list splits/hidsplat_v1/shapenet_test.txt \
--objaverse_eval_file_list splits/hidsplat_v1/objaverse_test.txt
```

Regenerate and validate the lists from the original run artifacts with:

```bash
python scripts/materialize_release_splits.py \
  --source-dir outputs/watermark_mixed_v2_dynamic_4k12k/splits \
  --output-dir ShapeSplat-Gaussian_MAE/splits/hidsplat_v1 \
  --test-manifest outputs/baseline_packages/manifest.json
```
