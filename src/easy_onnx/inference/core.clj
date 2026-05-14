(ns easy-onnx.inference.core
  "Internal helpers shared across easy-onnx.inference.* task wrappers.
  Not intended for direct consumer use; promoted to public from this namespace
  only so sibling task-wrapper namespaces can call into them."
  (:import [ai.onnxruntime OrtSession$SessionOptions OrtSession$SessionOptions$OptLevel]
           [io.github.inference4j.model LocalModelSource ModelSource]
           [io.github.inference4j.session SessionConfigurer]
           [java.nio.file Path]))

(def opt-level-map
  "Maps a Clojure keyword to an inference4j OrtSession$SessionOptions$OptLevel."
  {:none     OrtSession$SessionOptions$OptLevel/NO_OPT
   :basic    OrtSession$SessionOptions$OptLevel/BASIC_OPT
   :extended OrtSession$SessionOptions$OptLevel/EXTENDED_OPT
   :all      OrtSession$SessionOptions$OptLevel/ALL_OPT})

(defn ^SessionConfigurer ->session-configurer
  "Build a SessionConfigurer from a Clojure :session-options map.
  Recognizes :intra-op-num-threads, :inter-op-num-threads, :optimization-level."
  [{:keys [intra-op-num-threads inter-op-num-threads optimization-level]}]
  (reify SessionConfigurer
    (configure [_ opts]
      (let [opts ^OrtSession$SessionOptions opts]
        (when intra-op-num-threads
          (.setIntraOpNumThreads opts intra-op-num-threads))
        (when inter-op-num-threads
          (.setInterOpNumThreads opts inter-op-num-threads))
        (when optimization-level
          (.setOptimizationLevel opts (opt-level-map optimization-level)))))))

(defn ^ModelSource resolve-source
  "Pick a ModelSource from config keys :model-source / :base-dir.
  Returns nil to mean 'let inference4j use its default (HuggingFaceModelSource)'.
  :model-source takes precedence over :base-dir."
  [{:keys [model-source base-dir]}]
  (cond
    model-source model-source
    base-dir (LocalModelSource. (Path/of ^String base-dir (into-array String [])))
    :else nil))
