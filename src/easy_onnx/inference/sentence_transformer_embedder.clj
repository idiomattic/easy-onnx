(ns easy-onnx.inference.sentence-transformer-embedder
  (:require [com.stuartsierra.component :as component]
            [malli.core :as m])
  (:import [io.github.inference4j.nlp SentenceTransformerEmbedder]
           [java.lang AutoCloseable]))

(def Config
  (m/schema
   [:map
    [:model-id [:string {:min 1}]]]))

(defn- build-embedder
  ^SentenceTransformerEmbedder
  [{:keys [model-id]}]
  (-> (SentenceTransformerEmbedder/builder)
      (.modelId model-id)
      (.build)))

(defrecord Embedder [;; Config
                     model-id
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
  "Build and start a SentenceTransformerEmbedder.

  Required:
    :model-id - HuggingFace-style model id (e.g. \"inference4j/all-MiniLM-L6-v2\").
                On first call, inference4j downloads to ~/.cache/inference4j/."
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
