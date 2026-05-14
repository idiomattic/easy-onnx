(ns easy-onnx.inference.sentence-transformer-embedder
  (:require [com.stuartsierra.component :as component]
            [easy-onnx.inference.core :as core]
            [malli.core :as m])
  (:import [io.github.inference4j.nlp PoolingStrategy SentenceTransformerEmbedder]
           [java.lang AutoCloseable]))

(def Config
  (m/schema
   [:map
    [:model-id        [:string {:min 1}]]
    [:base-dir        {:optional true} [:string {:min 1}]]
    [:model-source    {:optional true} :any]
    [:pooling         {:optional true} [:enum :mean :cls :max]]
    [:normalize?      {:optional true} :boolean]
    [:text-prefix     {:optional true} [:string {:min 1}]]
    [:max-length      {:optional true} [:int {:min 1}]]
    [:session-options {:optional true}
     [:map
      [:intra-op-num-threads {:optional true} [:int {:min 1}]]
      [:inter-op-num-threads {:optional true} [:int {:min 1}]]
      [:optimization-level   {:optional true} [:enum :none :basic :extended :all]]]]]))

(def ^:private pooling-strategy-map
  {:mean PoolingStrategy/MEAN
   :cls  PoolingStrategy/CLS
   :max  PoolingStrategy/MAX})

(defn- build-embedder
  ^SentenceTransformerEmbedder
  [{:keys [model-id pooling normalize? text-prefix max-length session-options] :as config}]
  (let [builder (SentenceTransformerEmbedder/builder)]
    (.modelId builder model-id)
    (when-let [src (core/resolve-source config)]
      (.modelSource builder src))
    (when session-options
      (.sessionOptions builder (core/->session-configurer session-options)))
    (when pooling
      (.poolingStrategy builder (pooling-strategy-map pooling)))
    (when normalize?
      (.normalize builder))
    (when text-prefix
      (.textPrefix builder text-prefix))
    (when max-length
      (.maxLength builder max-length))
    (.build builder)))

(defrecord Embedder [;; Config
                     model-id
                     base-dir
                     model-source
                     pooling
                     normalize?
                     text-prefix
                     max-length
                     session-options

                     ;; Managed
                     embedder]
  AutoCloseable
  (close [_]
    (when embedder
      (SentenceTransformerEmbedder/.close ^SentenceTransformerEmbedder embedder)))

  component/Lifecycle
  (start [this]
    (if embedder
      this
      (assoc this :embedder (build-embedder this))))
  (stop [this]
    (.close this)
    (assoc this :embedder nil)))

(defn create
  "Build and start a SentenceTransformerEmbedder. Use with `with-open` for one-shot use.

  Required:
    :model-id - HuggingFace-style model id (e.g. \"inference4j/all-MiniLM-L6-v2\").
                On first call, inference4j downloads to ~/.cache/inference4j/.

  Optional:
    :base-dir    - Local directory containing model subdirectories. Switches to
                   LocalModelSource: <base-dir>/<model-id>/ must exist with model.onnx
                   and vocab.txt. Useful for offline use or custom model layouts.
    :model-source - Escape hatch: a Java io.github.inference4j.model.ModelSource
                    instance, used directly. Overrides :base-dir if both set.
    :pooling     - one of :mean :cls :max (default :mean).
    :normalize?  - L2-normalize the output (default false). Recommended when comparing
                   with cosine similarity (e.g. BGE, GTE, E5 families).
    :text-prefix - String prepended to every input before encoding. Required by some
                   model families (E5: \"query: \" / \"passage: \"; Nomic similar).
    :max-length  - Maximum token sequence length. Inputs longer than this are
                   truncated. Defaults to 512 (inference4j's default).
    :session-options - Map of ONNX Runtime session options:
                       :intra-op-num-threads - int. Threads within a single op.
                       :inter-op-num-threads - int. Threads parallelizing across ops.
                       :optimization-level   - one of :none :basic :extended :all."
  [config]
  {:pre [(m/validate Config config)]}
  (component/start (map->Embedder (assoc config :embedder nil))))

(defn component
  "Build an unstarted Embedder for use in a Stuart Sierra Component system.
  See `create` for accepted config keys."
  [config]
  {:pre [(m/validate Config config)]}
  (map->Embedder (assoc config :embedder nil)))

(defn close
  "Close the Embedder. Same as `(.close embedder)`."
  [^Embedder e]
  (.close e))

(defn encode
  "Encode `text` into a float[] embedding."
  [{:keys [^SentenceTransformerEmbedder embedder]} ^String text]
  (SentenceTransformerEmbedder/.encode embedder text))

(defn encode-batch
  "Encode each text in `texts` into a float[] embedding. Returns a vector of float[]."
  [{:keys [^SentenceTransformerEmbedder embedder]} texts]
  (vec (SentenceTransformerEmbedder/.encodeBatch embedder texts)))
