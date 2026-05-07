(ns easy-onnx.runtime
  (:require [com.stuartsierra.component :refer [Lifecycle start]]
            [malli.core :as m])
  (:import [ai.onnxruntime OnnxTensor OnnxValue OrtEnvironment OrtSession OrtSession$Result]
           [java.lang AutoCloseable]))

;; NOTE: Reconsider malli-vs-plain-validation if config schemas stay trivial
;; after the API stabilizes. See spec, Future considerations.
(def Config
  (m/schema
   [:map
    [:model-path [:string {:min 1}]]]))

(defrecord Session [model-path env session]
  AutoCloseable
  (close [_]
    (when session
      (OrtSession/.close ^OrtSession session)))

  Lifecycle
  (start [this]
    (if session
      this
      (let [env (OrtEnvironment/getEnvironment)
            sess (OrtEnvironment/.createSession env ^String model-path)]
        (assoc this :env env :session sess))))
  (stop [this]
    (.close this)
    (assoc this :env nil :session nil)))

(defn create
  "Build and start a Session. Use with `with-open` for one-shot use."
  [config]
  {:pre [(m/validate Config config)]}
  (start (->Session (:model-path config) nil nil)))

(defn component
  "Build an unstarted Session for use in a Stuart Sierra Component system."
  [config]
  {:pre [(m/validate Config config)]}
  (->Session (:model-path config) nil nil))

(defn close
  "Close the Session. Same as `(.close session)`."
  [^Session session]
  (.close session))

(defn- ->tensors
  "Convert {input-name -> Java array} to {input-name -> OnnxTensor}."
  [^OrtEnvironment env inputs]
  (reduce-kv (fn [m k v]
               (assoc m k (OnnxTensor/createTensor env v)))
             {}
             inputs))

(defn- result->map
  "Extract all outputs from an OrtSession$Result into {output-name -> value}."
  [^OrtSession$Result result]
  (into {}
        (map (fn [^java.util.Map$Entry entry]
               [(.getKey entry)
                (-> ^OnnxValue (.getValue entry) OnnxValue/.getValue)]))
        result))

(defn run-model
  "Run inference on `inputs`, a map of {input-name -> Java array}.
  Returns {output-name -> raw value}. Manages OnnxTensor and Result lifecycles."
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
