# Tiny-GenImage raw download

This folder stores the original downloaded files. Do not edit images in place.

## Source

- Hugging Face: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
- Upstream Kaggle tiny subset: https://www.kaggle.com/datasets/yangsangtai/tiny-genimage
- Original benchmark: Zhu et al., GenImage (CC BY-NC-SA 4.0)

## What we downloaded

Not the full GenImage dataset (~500-650 GB, more than 2 million images).

We downloaded two Tiny-GenImage **train** parquet shards:

- `data/train-00000-of-00014.parquet` (~477 MB, 2,000 images)
- `data/train-00001-of-00014.parquet` (~475 MB, 2,000 images)

That is about **4,000 images** and about **0.95 GB** of parquet files.

The remaining 16 shards (about 7.4 GB) were not downloaded.

## Why two shards

One shard is already a 2,000-image balanced pool. A second shard brings the
pilot to the planned 2,000-4,000 range and gives more images per generator
before we create source-aware splits.
