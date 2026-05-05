(ns easy-onnx.onnx
  (:require [malli.core :as m]
            [mokujin.log :as log]
            [utility-belt.component :as util.component])
  (:import [ai.onnxruntime OnnxTensor OnnxValue OrtEnvironment OrtSession OrtSession$Result]))

(def OnnxModelConfig
  (m/schema
   [:map
    [:model-path
     [:string {:min 1}]]]))

(defn create
  [{:keys [model-path] :as config}]
  {:pre [(m/validate OnnxModelConfig config)]}
  (util.component/map->component
   {:init {:model-path model-path}
    :start (fn [this]
             (if (:session this)
               this
               (try
                 (let [env (OrtEnvironment/getEnvironment)
                       session (OrtEnvironment/.createSession env ^String model-path)]
                   (assoc this
                          :env env
                          :session session))
                 (catch Throwable e
                   (log/error e "could not initialize model" {:model-path model-path})))))
    :end (fn [this]
           (when-let [s (:session this)]
             (OrtSession/.close ^OrtSession s))
           (assoc this :session nil :env nil))}))

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
