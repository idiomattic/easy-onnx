# easy-onnx

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.idiomattic/easy-onnx.svg)](https://clojars.org/net.clojars.idiomattic/easy-onnx)
[![CI](https://github.com/idiomattic/easy-onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/idiomattic/easy-onnx/actions/workflows/ci.yml)

A Clojure library for sentence-transformer text embedding and small-model
text generation (via [inference4j](https://github.com/inference4j/inference4j))
plus embedding analysis (DBSCAN clustering, UMAP projection, cosine similarity
via [smile-core](https://github.com/haifengl/smile)).

> [!WARNING]
> This library should be considered pre-release. Both top-level sections
> (`easy-onnx.inference` and `easy-onnx.analysis`) are designed to be
> extracted into separate libraries later.

## Installation

Add to `deps.edn`:

```clojure
net.clojars.idiomattic/easy-onnx {:mvn/version "0.3.XXX"}
```

## Quickstart

```clojure
(require '[easy-onnx.inference.sentence-transformer-embedder :as ste]
         '[easy-onnx.analysis :as analysis])

(with-open [e (ste/create {:model-id "inference4j/all-MiniLM-L6-v2"
                           :normalize? true})]
  (let [v1 (ste/encode e "hello world")
        v2 (ste/encode e "hi there")]
    (analysis/cosine-similarity v1 v2)))
```

The first call to `ste/create` downloads the model (~80MB for MiniLM) to
`~/.cache/inference4j/inference4j/all-MiniLM-L6-v2/`. Subsequent calls use
the cache.

## Sections

### `easy-onnx.inference.sentence-transformer-embedder`

Wraps [inference4j's `SentenceTransformerEmbedder`](https://github.com/inference4j/inference4j).
Compatible with all-MiniLM, all-mpnet, BGE, GTE, and E5 family models.

```clojure
(ste/create {:model-id "inference4j/all-MiniLM-L6-v2"
             :pooling :mean         ;; :mean | :cls | :max     (default :mean)
             :normalize? true       ;; L2-normalize the output  (default false)
             :text-prefix "query: " ;; E5/Nomic prefix support (optional)
             :max-length 512})      ;; truncation length        (default 512)
;; => Embedder (started, AutoCloseable + Lifecycle)

(ste/encode embedder "text")        ;; => float[]
(ste/encode-batch embedder ["a" "b"]) ;; => vector of float[]
(ste/close embedder)                ;; idempotent
```

#### Model loading

- **Default** (`:model-id` only): auto-download from HuggingFace to
  `~/.cache/inference4j/`. Models hosted at
  [huggingface.co/inference4j](https://huggingface.co/inference4j).
- **Local** (`:base-dir "/some/dir" :model-id "subdir-name"`): load from
  `<base-dir>/<model-id>/` via `LocalModelSource`.
- **Custom** (`:model-source <ModelSource-instance>`): bring your own
  `io.github.inference4j.model.ModelSource`.

#### Component integration

`ste/component` returns an unstarted `Embedder` for use in a Stuart Sierra
Component system:

```clojure
(component/system-map
  :embedder (ste/component {:model-id "inference4j/all-MiniLM-L6-v2"}))
```

### `easy-onnx.inference.onnx-text-generator`

Wraps [inference4j's `OnnxTextGenerator`](https://github.com/inference4j/inference4j).
Six preset model families. Supports blocking and streaming generation.

```clojure
(require '[easy-onnx.inference.onnx-text-generator :as otg])

;; one-shot blocking
(with-open [g (otg/create {:preset :qwen2
                           :max-new-tokens 100
                           :temperature 0.7
                           :top-k 50})]
  (-> (otg/generate g "Explain gravity")
      :text
      println))

;; streaming
(with-open [g (otg/create {:preset :qwen2 :max-new-tokens 100})]
  (otg/generate-streaming g "Explain gravity"
                          (fn [token] (print token) (flush))))
```

`generate` and `generate-streaming` both return `{:text :prompt-tokens
:generated-tokens :duration}`. `:duration` is a `java.time.Duration`.

#### Presets

| Preset            | Model                                       | Approx. cached size |
| ----------------- | ------------------------------------------- | ------------------- |
| `:gpt2`           | GPT-2 124M (completion)                     | ~500 MB             |
| `:smol-lm-2`      | SmolLM2-360M-Instruct (ChatML)              | ~700 MB             |
| `:smol-lm-2-1-7b` | SmolLM2-1.7B-Instruct (FP16)                | ~3.4 GB             |
| `:qwen2`          | Qwen2.5-1.5B-Instruct (ChatML)              | ~3 GB               |
| `:tiny-llama`     | TinyLlama-1.1B-Chat                         | ~2.2 GB             |
| `:gemma2`         | Gemma 2-2B-IT (gated; bring your own files) | —                   |

Each preset bundles its model id, special tokens, stop sequences, chat
template, and tokenizer. The first call downloads to
`~/.cache/inference4j/`; subsequent calls hit the cache. Gemma 2 is gated
on HuggingFace, so you must accept Google's license, download the ONNX
model yourself, and provide it via `:base-dir` or `:model-source`.

#### Sampling

```clojure
(otg/create {:preset :qwen2
             :max-new-tokens 100 ;; default 256
             :temperature 0.7    ;; 0.0 = greedy
             :top-k 50           ;; 0 = disabled
             :top-p 0.9})        ;; 0.0 = disabled
```

#### Model loading

- **Default** (`:preset` only): auto-download from HuggingFace.
- **Local** (`:base-dir "/some/dir"`): load from
  `<base-dir>/<preset-model-id>/` via `LocalModelSource`. Useful for
  air-gapped environments or gated models.
- **Custom** (`:model-source <ModelSource-instance>`): bring your own
  `io.github.inference4j.model.ModelSource`.

#### Component integration

```clojure
(component/system-map
  :generator (otg/component {:preset :qwen2}))
```

### `easy-onnx.analysis`

Smile-based analysis on raw `float[]` vectors:

```clojure
(analysis/cosine-similarity v1 v2)  ;; ^double in [-1, 1]
(analysis/cosine-distance v1 v2)    ;; ^double in [0, 2]
(analysis/cluster embeddings)       ;; DBSCAN; vector of cluster indices
(analysis/project-2d embeddings)    ;; UMAP; vector of [x y] pairs
```

## JVM flags

ONNX Runtime and Smile both use `System.load` to load native libraries.
On Java 24+, the JVM emits a "restricted method has been called" warning
for these calls and may block them in a future release. To silence the
warning today and future-proof your app, add `--enable-native-access=ALL-UNNAMED`
to your JVM options. For Clojure CLI projects:

```clojure
;; deps.edn
{:aliases {:run {:jvm-opts ["--enable-native-access=ALL-UNNAMED"]}}}
```

## Tests

```bash
clojure -M:test
```

The first run downloads MiniLM to `~/.cache/inference4j/`. Subsequent runs
are fast (cache hit). No manual fixture download is required.

## License

Copyright © 2026 Matthew Lese

Distributed under the Eclipse Public License version 1.0.
