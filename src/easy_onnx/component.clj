(ns easy-onnx.component
  "WIP: exploring whether a unified component makes sense"
  (:require [malli.core :as m]
            [mokujin.log :as log]
            [utility-belt.component :as util.component])
  (:import [ai.djl.huggingface.tokenizers Encoding HuggingFaceTokenizer]
           [ai.onnxruntime OnnxTensor OnnxValue OrtEnvironment OrtSession OrtSession$Result]
           [java.nio.file Path]))

(def OnnxConfig
  (m/schema
   [:map
    [:tokenizer-path
     [:string {:min 1}]]
    [:model-path
     [:string {:min 1}]]]))

(defn with-session [this model-path]
  (if (:session this)
    this
    (try
      (let [env (OrtEnvironment/getEnvironment)
            session (OrtEnvironment/.createSession env ^String model-path)]
        (assoc this
               :env env
               :session session))
      (catch Throwable e
        (log/error e "could not initialize model" {:model-path model-path})
        this))))

(defn with-tokenizer [this tokenizer-path]
  (if (:tokenizer this)
    this
    (try
      (let [path (Path/of ^String tokenizer-path (into-array String []))
            tokenizer (HuggingFaceTokenizer/newInstance path)]
        (assoc this :tokenizer tokenizer))
      (catch Throwable e
        (log/error e "could not initialize tokenizer" {:tokenizer-path tokenizer-path})
        this))))

(defn create
  [{:keys [tokenizer-path model-path] :as config}]
  {:pre [(m/validate OnnxConfig config)]}
  (util.component/map->component
   {:init {:tokenizer-path tokenizer-path :model-path model-path}
    :start (fn [this]
             (-> this
                 (with-session model-path)
                 (with-tokenizer tokenizer-path)))
    :end (fn [this]
           (when-let [s (:session this)]
             (OrtSession/.close ^OrtSession s))
           (when-let [^HuggingFaceTokenizer t (:tokenizer this)]
             (HuggingFaceTokenizer/.close t))
           (assoc this :session nil :env nil :tokenizer nil))}))

;; ---------------------------------------------------------------------------
;; Client utils
;; ---------------------------------------------------------------------------

(defn encode!
  [{:keys [^HuggingFaceTokenizer tokenizer]} ^String text]
  (HuggingFaceTokenizer/.encode tokenizer text))

(defn- ->tensors
  "Convert a map of {name → raw-data} to {name → OnnxTensor}.
  raw-data should be a Java array (long[][], float[][], etc.)."
  [^OrtEnvironment env inputs]
  (reduce-kv (fn [m k v]
               (assoc m k (OnnxTensor/createTensor env v)))
             {}
             inputs))

(defn- result->map
  "Extract all outputs from an OrtSession.Result into a Clojure map
  of {output-name → value}."
  [^OrtSession$Result result]
  (into {}
        (map (fn [^java.util.Map$Entry entry]
               [(.getKey entry) (-> ^OnnxValue (.getValue entry) OnnxValue/.getValue)]))
        result))

(defn run-model!
  [{:keys [^OrtEnvironment env ^OrtSession session]} inputs]
  (let [tensors (->tensors env inputs)]
    (try
      (let [result (OrtSession/.run session tensors)]
        (try
          (result->map result)
          (finally
            (OrtSession$Result/.close result))))
      (finally
        (run! (fn [[_ ^OnnxTensor t]] (OnnxTensor/.close t)) tensors)))))
