(ns easy-onnx.runtime
  (:require [com.stuartsierra.component :as component]
            [malli.core :as m])
  (:import [ai.onnxruntime OnnxModelMetadata OnnxTensor OnnxValue OrtEnvironment OrtLoggingLevel
            OrtSession OrtSession$Result OrtSession$SessionOptions
            OrtSession$SessionOptions$OptLevel]
           [java.lang AutoCloseable]))

;; NOTE: Reconsider malli-vs-plain-validation if config schemas stay trivial
;; after the API stabilizes. See spec, Future considerations.
(def Config
  (m/schema
   [:map
    [:model-path [:string {:min 1}]]
    [:intra-op-num-threads {:optional true} [:int {:min 1}]]
    [:inter-op-num-threads {:optional true} [:int {:min 1}]]
    [:optimization-level {:optional true} [:enum :none :basic :extended :all]]
    [:log-level {:optional true} [:enum :verbose :info :warning :error :fatal]]]))

(def ^:private opt-level-map
  {:none OrtSession$SessionOptions$OptLevel/NO_OPT
   :basic OrtSession$SessionOptions$OptLevel/BASIC_OPT
   :extended OrtSession$SessionOptions$OptLevel/EXTENDED_OPT
   :all OrtSession$SessionOptions$OptLevel/ALL_OPT})

(def ^:private log-level-map
  {:verbose OrtLoggingLevel/ORT_LOGGING_LEVEL_VERBOSE
   :info OrtLoggingLevel/ORT_LOGGING_LEVEL_INFO
   :warning OrtLoggingLevel/ORT_LOGGING_LEVEL_WARNING
   :error OrtLoggingLevel/ORT_LOGGING_LEVEL_ERROR
   :fatal OrtLoggingLevel/ORT_LOGGING_LEVEL_FATAL})

(defn- ->session-options
  "Build OrtSession.SessionOptions from a config map. Caller must close."
  ^OrtSession$SessionOptions [{:keys [intra-op-num-threads inter-op-num-threads optimization-level log-level]}]
  (let [opts (OrtSession$SessionOptions.)]
    (when intra-op-num-threads
      (OrtSession$SessionOptions/.setIntraOpNumThreads opts intra-op-num-threads))
    (when inter-op-num-threads
      (OrtSession$SessionOptions/.setInterOpNumThreads opts inter-op-num-threads))
    (when optimization-level
      (OrtSession$SessionOptions/.setOptimizationLevel opts (opt-level-map optimization-level)))
    (when log-level
      (OrtSession$SessionOptions/.setSessionLogLevel opts (log-level-map log-level)))
    opts))

(defn- build-session
  "Build an OrtSession from the env and config options.
  Closes the SessionOptions before returning."
  ^OrtSession [^OrtEnvironment env config]
  (let [opts (->session-options config)]
    (try
      (OrtEnvironment/.createSession env ^String (:model-path config) opts)
      (finally
        (OrtSession$SessionOptions/.close opts)))))

(defrecord Session [;; Configuration
                    model-path
                    intra-op-num-threads
                    inter-op-num-threads
                    optimization-level
                    log-level

                    ;; Managed
                    env
                    session]
  AutoCloseable
  (close [_]
    (when session
      (OrtSession/.close ^OrtSession session)))

  component/Lifecycle
  (start [this]
    (if session
      this
      (let [env (OrtEnvironment/getEnvironment)
            sess (build-session env this)]
        (assoc this :env env :session sess))))
  (stop [this]
    (.close this)
    (assoc this :env nil :session nil)))

(defn create
  "Build and start a Session. Use with `with-open` for one-shot use.

  Required:
    :model-path - Path to an ONNX model file.

  Optional session settings:
    :intra-op-num-threads - int. Threads used inside a single op.
                            Defaults to all available cores; tune to your
                            container's CPU limit to avoid throttling.
    :inter-op-num-threads - int. Threads used to parallelize across ops.
    :optimization-level   - one of :none :basic :extended :all (default :all).
    :log-level            - one of :verbose :info :warning :error :fatal."
  [config]
  {:pre [(m/validate Config config)]}
  (component/start (map->Session (assoc config :env nil :session nil))))

(defn component
  "Build an unstarted Session for use in a Stuart Sierra Component system.
  See `create` for accepted config keys."
  [config]
  {:pre [(m/validate Config config)]}
  (map->Session (assoc config :env nil :session nil)))

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

(defn metadata
  "Return the loaded ONNX model's metadata as a Clojure map.

  Keys:
    :producer-name     - tool that produced the model (e.g. \"pytorch\")
    :graph-name        - graph name embedded in the model
    :graph-description - graph description (often empty)
    :domain            - model domain (often empty)
    :description       - model description (often empty)
    :version           - model version. Long.MAX_VALUE means \"unset\".
    :custom-metadata   - {String String} map of any custom tags the
                         exporter attached. Empty for most exports."
  [{:keys [^OrtSession session]}]
  (let [m (OrtSession/.getMetadata session)]
    {:producer-name (OnnxModelMetadata/.getProducerName m)
     :graph-name (OnnxModelMetadata/.getGraphName m)
     :graph-description (OnnxModelMetadata/.getGraphDescription m)
     :domain (OnnxModelMetadata/.getDomain m)
     :description (OnnxModelMetadata/.getDescription m)
     :version (OnnxModelMetadata/.getVersion m)
     :custom-metadata (into {} (OnnxModelMetadata/.getCustomMetadata m))}))
