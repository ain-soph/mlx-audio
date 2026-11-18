# Provenance

CosyVoice3 (with the CosyVoice2 modules it reuses) and the S3 speech tokenizer / S3Gen codec
(`mlx_audio/codec/models/s3tokenizer`, `mlx_audio/codec/models/s3gen`) are ported unchanged from
mlx-audio-plus (https://github.com/DePasqualeOrg/mlx-audio-plus, commit 4c9ec6a), MIT License,
Copyright © Anthony DePasquale, Copyright © 2024 Prince Canuma.

Changes in this port:
- `mlx_lm` imports (KVCache, the Qwen2 model) use the copies vendored in `mlx_audio.lm`.
- Runs with transformers 5 (mlx-audio-plus pins transformers < 5).
