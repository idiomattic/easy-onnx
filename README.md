# easy-onnx

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.idiomattic/easy-onnx.svg)](https://clojars.org/net.clojars.idiomattic/easy-onnx)
[![CI](https://github.com/idiomattic/easy-onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/idiomattic/easy-onnx/actions/workflows/ci.yml)

A Clojure library for sentence-transformer text embedding (via
[inference4j](https://github.com/inference4j/inference4j)) plus embedding
analysis (DBSCAN clustering, UMAP projection, cosine similarity via
[smile-core](https://github.com/haifengl/smile)).

> [!WARNING]
> This library should be considered pre-release. Both top-level sections
> (`easy-onnx.inference` and `easy-onnx.analysis`) are designed to be
> extracted into separate libraries later.

## Installation

Add to `deps.edn`:

```clojure
net.clojars.idiomattic/easy-onnx {:mvn/version "0.2.XXX"}
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
(ste/create {:model-id     "inference4j/all-MiniLM-L6-v2"
             :pooling      :mean    ;; :mean | :cls | :max     (default :mean)
             :normalize?   true     ;; L2-normalize the output  (default false)
             :text-prefix  "query: " ;; E5/Nomic prefix support (optional)
             :max-length   512})    ;; truncation length        (default 512)
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
