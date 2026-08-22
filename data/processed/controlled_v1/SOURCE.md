# controlled_v1 processed images

This folder contains a **bias-mitigated** copy of the Tiny-GenImage pilot.

It is not unbiased, bias-free, or leakage-free.

Raw originals remain untouched under `data/raw/`.

Processing version: controlled_v1
- convert to RGB
- resize shortest side to 256 (LANCZOS)
- centre crop 224×224
- JPEG quality 96, subsampling 0
