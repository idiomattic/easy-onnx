(ns easy-onnx.inference.onnx-text-generator
  (:require [com.stuartsierra.component :as component]
            [malli.core :as m])
  (:import [io.github.inference4j.generation GenerationResult]
           [io.github.inference4j.nlp OnnxTextGenerator]
           [java.lang AutoCloseable]))

(def Config
  (m/schema
   [:map
    [:preset [:enum :gpt2 :smol-lm-2 :smol-lm-2-1-7b :qwen2 :tiny-llama :gemma2]]]))

(def ^:private preset-factory-map
  {:gpt2            #(OnnxTextGenerator/gpt2)
   :smol-lm-2       #(OnnxTextGenerator/smolLM2)
   :smol-lm-2-1-7b  #(OnnxTextGenerator/smolLM2_1_7B)
   :qwen2           #(OnnxTextGenerator/qwen2)
   :tiny-llama      #(OnnxTextGenerator/tinyLlama)
   :gemma2          #(OnnxTextGenerator/gemma2)})

(defn- ^OnnxTextGenerator build-generator
  [{:keys [preset]}]
  (let [builder ((preset-factory-map preset))]
    (.build builder)))

(defn- ->result-map [^GenerationResult r]
  {:text             (.text r)
   :prompt-tokens    (.promptTokens r)
   :generated-tokens (.generatedTokens r)
   :duration         (.duration r)})

(defrecord Generator [;; Config
                      preset

                      ;; Managed
                      generator]
  AutoCloseable
  (close [_]
    (when generator
      (OnnxTextGenerator/.close ^OnnxTextGenerator generator)))

  component/Lifecycle
  (start [this]
    (if generator
      this
      (assoc this :generator (build-generator this))))
  (stop [this]
    (.close this)
    (assoc this :generator nil)))

(defn create
  "Build and start an OnnxTextGenerator. Use with `with-open` for one-shot use.

  Required:
    :preset - one of :gpt2 :smol-lm-2 :smol-lm-2-1-7b :qwen2 :tiny-llama :gemma2.
              On first call, inference4j downloads to ~/.cache/inference4j/."
  [config]
  {:pre [(m/validate Config config)]}
  (component/start (map->Generator (assoc config :generator nil))))

(defn component
  "Build an unstarted Generator for use in a Stuart Sierra Component system.
  See `create` for accepted config keys."
  [config]
  {:pre [(m/validate Config config)]}
  (map->Generator (assoc config :generator nil)))

(defn close
  "Close the Generator. Same as `(.close generator)`."
  [^Generator g]
  (.close g))

(defn generate
  "Generate text for `prompt`. Blocks until generation completes.
  Returns {:text :prompt-tokens :generated-tokens :duration}."
  [{:keys [^OnnxTextGenerator generator]} ^String prompt]
  (->result-map (OnnxTextGenerator/.generate generator prompt)))
