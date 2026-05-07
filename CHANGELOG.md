# Change Log

All notable changes to this project will be documented in this file.
This change log follows the conventions of
[keepachangelog.com](http://keepachangelog.com/).

## [Unreleased]

### Added
- `easy-onnx.runtime` — ONNX session wrapper with `create` / `component` /
  `run-model` / `close`. AutoCloseable + Stuart Sierra Lifecycle.
- `easy-onnx.tokenizer` — HuggingFace tokenizer wrapper with `create` /
  `component` / `encode` / `get-ids` / `get-mask` / `close`.
- `easy-onnx.embed` — `cosine-similarity` and `cosine-distance` over float[].
- `easy-onnx.embed.text` — text embedding via mean-pooling (sentence-
  transformer-shaped models).
- `easy-onnx.analysis` — DBSCAN clustering and UMAP 2D projection over
  float[] embeddings.
