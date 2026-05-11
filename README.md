# easy-onnx

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.idiomattic/easy-onnx.svg)](https://clojars.org/net.clojars.idiomattic/easy-onnx)
[![CI](https://github.com/idiomattic/easy-onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/idiomattic/easy-onnx/actions/workflows/ci.yml)

A Clojure library for running ONNX models and tokenizing text with
HuggingFace tokenizers, plus opinionated helpers for sentence-transformer
text embedding and embedding analysis (DBSCAN clustering, UMAP projection).

## Status

Pre-release. The library is internally split into four sections — `runtime`,
`tokenizer`, `embed`, `analysis` — designed for future extraction into
separate artifacts if needed.

## Installation

Add to `deps.edn`:

```clojure
net.clojars.easy-onnx/easy-onnx {:mvn/version "0.1.0-SNAPSHOT"}
```

(Maven coordinates are tentative; will be confirmed before first release.)

## Quickstart

```clojure
(require '[easy-onnx.runtime :as runtime]
         '[easy-onnx.tokenizer :as tokenizer]
         '[easy-onnx.embed.text :as embed-text]
         '[easy-onnx.embed :as embed])

(with-open [r (runtime/create   {:model-path "path/to/model.onnx"})
            t (tokenizer/create {:tokenizer-path "path/to/tokenizer.json"})]
  (let [v1 (embed-text/embed {:runtime r :tokenizer t} "hello world")
        v2 (embed-text/embed {:runtime r :tokenizer t} "hi there")]
    (embed/cosine-similarity v1 v2)))
```

## Sections

- **`easy-onnx.runtime`** — load and run ONNX models. Wraps `OrtEnvironment`
  - `OrtSession`. `create` returns a started `Session`; `component` returns
    an unstarted one for use with Stuart Sierra Component. Both implement
    `AutoCloseable` and Lifecycle.
- **`easy-onnx.tokenizer`** — wrap a HuggingFace tokenizer. Same shape:
  `create`, `component`, `encode`, `get-ids`, `get-mask`, `close`.
- **`easy-onnx.embed`** — modality-agnostic vector helpers: `cosine-similarity`,
  `cosine-distance`.
- **`easy-onnx.embed.text`** — text-specific embedding via mean-pooling.
  Currently sentence-transformer-shaped (e.g., MiniLM).
- **`easy-onnx.analysis`** — embedding analysis: DBSCAN clustering, UMAP
  2D projection.

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

Tests for `runtime`, `tokenizer`, and `embed.text` require the MiniLM
fixture at `resources/ml/all-MiniLM-L6-v2/`. Tests skip gracefully when
the fixture is absent.

## License

Copyright © 2026 Matthew Lese

Distributed under the Eclipse Public License version 1.0.
