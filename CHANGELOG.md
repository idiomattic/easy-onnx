# Change Log

All notable changes to this project will be documented in this file.
This change log follows the conventions of
[keepachangelog.com](http://keepachangelog.com/).

## 0.3.0 — 2026-05-14

### Added

- `easy-onnx.inference.onnx-text-generator` — wraps inference4j's `OnnxTextGenerator` for autoregressive text generation. Six presets: `:gpt2`, `:smol-lm-2`, `:smol-lm-2-1-7b`, `:qwen2`, `:tiny-llama`, `:gemma2`. Two verbs: `generate` (blocking) and `generate-streaming` (callback-based). Both return `{:text :prompt-tokens :generated-tokens :duration}`.
- Sampling config keys: `:max-new-tokens`, `:temperature`, `:top-k`, `:top-p`.
- `easy-onnx.inference.core` — internal namespace holding shared helpers (`opt-level-map`, `->session-configurer`, `resolve-source`) used by every `inference` task wrapper.

### Changed

- `easy-onnx.inference.sentence-transformer-embedder` now delegates session-options handling and model-source resolution to `easy-onnx.inference.core`. Public API and behavior are unchanged.

## 0.2.0 — 2026-05-14

### Changed

- Swapped `com.microsoft.onnxruntime/onnxruntime` and `ai.djl.huggingface/tokenizers` for `io.github.inference4j/inference4j-core 0.10.0`. inference4j wraps both ONNX Runtime and HuggingFace tokenizers and provides idiomatic per-task wrappers (sentence-transformer embedding, classification, reranking, etc.).
- The public API contracted from five namespaces to two:
  - **New** `easy-onnx.inference.sentence-transformer-embedder` (replaces `runtime` + `tokenizer` + `embed.text`).
  - **Updated** `easy-onnx.analysis` (now also exposes `cosine-similarity` / `cosine-distance`, lifted from the deleted `easy-onnx.embed`).
- Model loading is now automatic. The first call to `ste/create` downloads to `~/.cache/inference4j/`. No more manual `resources/ml/` fixture.
- Embedder gains `:pooling` (`:mean`/`:cls`/`:max`), `:normalize?` (L2), `:text-prefix` (E5/Nomic support), and `:max-length` config keys.

### Removed

- `easy-onnx.runtime`, `easy-onnx.tokenizer`, `easy-onnx.embed`, and `easy-onnx.embed.text` namespaces (and their tests).
- `:log-level` config key (not exposed by inference4j's task wrappers).
- The manual MiniLM fixture in `resources/ml/`. inference4j manages model caching.

### Migration

Before:

```clojure
(with-open [r (runtime/create   {:model-path "/path/to/model.onnx"})
            t (tokenizer/create {:tokenizer-path "/path/to/tokenizer.json"})]
  (embed-text/embed {:runtime r :tokenizer t} "hello world"))
```

After:

```clojure
(with-open [e (ste/create {:model-id "inference4j/all-MiniLM-L6-v2"})]
  (ste/encode e "hello world"))
```

## 0.1.0

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
