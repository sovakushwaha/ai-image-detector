# V2-8 Kaggle CLIP LoRA (private)

Parameter-efficient CLIP ViT-B/16 adaptation (LoRA on last 4 visual blocks) + MLP-B head.

## Modes

Set `run_config.json` before push:

```json
{"mode": "smoke"}
```

or

```json
{"mode": "full"}
```

## Smoke test

- Fold 1 only
- Up to 128 Real + 128 AI train; 2 epochs max
- Engineering gate only

## Full training

- Four locked V2-3 folds
- 20 epochs max, early stopping patience 4
- Outputs under `/kaggle/working/v2_lora_outputs/`

## Data policy

- Tiny-GenImage: Kaggle dataset `yangsangtai/tiny-genimage` + pinned paths
- MLLM / Qwen: Hugging Face pinned revisions (internet enabled)
- COCO: official val2017 URLs
- Smartphone: reconstruct from locked manifest URLs (not republished)

## NTIRE

Not accessed.

## Credentials

Never place API tokens in this directory.
