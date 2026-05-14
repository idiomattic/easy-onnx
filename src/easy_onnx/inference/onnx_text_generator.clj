(ns easy-onnx.inference.onnx-text-generator
  (:require [com.stuartsierra.component :as component]
            [easy-onnx.inference.core :as core]
            [malli.core :as m])
  (:import [io.github.inference4j.generation GenerationResult]
           [io.github.inference4j.nlp OnnxTextGenerator]
           [java.lang AutoCloseable]
           [java.util.function Consumer]))

(def Config
  (m/schema
   [:map
    [:preset [:enum :gpt2 :smol-lm-2 :smol-lm-2-1-7b :qwen2 :tiny-llama :gemma2]]
    [:max-new-tokens {:optional true} [:int {:min 1}]]
    [:temperature {:optional true} [:double {:min 0.0}]]
    [:top-k {:optional true} [:int {:min 0}]]
    [:top-p {:optional true} [:double {:min 0.0 :max 1.0}]]
    [:base-dir {:optional true} [:string {:min 1}]]
    [:model-source {:optional true} :any]
    [:session-options {:optional true}
     [:map
      [:intra-op-num-threads {:optional true} [:int {:min 1}]]
      [:inter-op-num-threads {:optional true} [:int {:min 1}]]
      [:optimization-level {:optional true} [:enum :none :basic :extended :all]]]]]))

(def ^:private preset-factory-map
  {:gpt2 #(OnnxTextGenerator/gpt2)
   :smol-lm-2 #(OnnxTextGenerator/smolLM2)
   :smol-lm-2-1-7b #(OnnxTextGenerator/smolLM2_1_7B)
   :qwen2 #(OnnxTextGenerator/qwen2)
   :tiny-llama #(OnnxTextGenerator/tinyLlama)
   :gemma2 #(OnnxTextGenerator/gemma2)})

(defn- build-generator
  ^OnnxTextGenerator [{:keys [preset max-new-tokens temperature top-k top-p session-options] :as config}]
  (let [builder ((preset-factory-map preset))]
    (when-let [src (core/resolve-source config)]
      (.modelSource builder src))
    (when session-options
      (.sessionOptions builder (core/->session-configurer session-options)))
    (when max-new-tokens
      (.maxNewTokens builder (int max-new-tokens)))
    (when temperature
      (.temperature builder (float temperature)))
    (when top-k
      (.topK builder (int top-k)))
    (when top-p
      (.topP builder (float top-p)))
    (.build builder)))

(defn- ->result-map [^GenerationResult r]
  {:text (.text r)
   :prompt-tokens (.promptTokens r)
   :generated-tokens (.generatedTokens r)
   :duration (.duration r)})

(defrecord Generator [;; Config
                      preset
                      max-new-tokens
                      temperature
                      top-k
                      top-p
                      base-dir
                      model-source
                      session-options

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
              On first call, inference4j downloads to ~/.cache/inference4j/.

  Optional:
    :max-new-tokens - maximum number of tokens to generate (default 256).
    :temperature    - sampling temperature in [0.0, ...). 0.0 = greedy decoding.
    :top-k          - keep the top-K candidate tokens; 0 disables (default 0).
    :top-p          - keep tokens whose cumulative probability reaches top-P;
                      0.0 disables (default 0.0).
    :base-dir       - Local directory containing model subdirectories. Switches to
                      LocalModelSource: <base-dir>/<preset-model-id>/ must exist.
                      Useful for offline use or custom model layouts.
    :model-source   - Escape hatch: a Java io.github.inference4j.model.ModelSource
                      instance, used directly. Overrides :base-dir if both set.
    :session-options - Map of ONNX Runtime session options:
                       :intra-op-num-threads - int. Threads within a single op.
                       :inter-op-num-threads - int. Threads parallelizing across ops.
                       :optimization-level   - one of :none :basic :extended :all."
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

(defn generate-streaming
  "Generate text for `prompt`, calling `on-token` (a 1-arg fn) with each token
  as it's produced. Returns the same map shape as `generate`."
  [{:keys [^OnnxTextGenerator generator]} ^String prompt on-token]
  (let [consumer (reify Consumer
                   (accept [_ token] (on-token token)))]
    (->result-map (OnnxTextGenerator/.generate generator prompt consumer))))
