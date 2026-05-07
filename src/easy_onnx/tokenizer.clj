(ns easy-onnx.tokenizer
  (:require [com.stuartsierra.component :refer [Lifecycle start]]
            [malli.core :as m])
  (:import [ai.djl.huggingface.tokenizers Encoding HuggingFaceTokenizer]
           [java.lang AutoCloseable]
           [java.nio.file Path]))

;; NOTE: Reconsider malli-vs-plain-validation if config schemas stay trivial
;; after the API stabilizes. See spec, Future considerations.
(def Config
  (m/schema
   [:map
    [:tokenizer-path [:string {:min 1}]]]))

(defrecord Tokenizer [;; Configuration
                      tokenizer-path

                      ;; Managed
                      tokenizer]
  AutoCloseable
  (close [_]
    (when tokenizer
      (HuggingFaceTokenizer/.close ^HuggingFaceTokenizer tokenizer)))

  Lifecycle
  (start [this]
    (if tokenizer
      this
      (let [path (Path/of ^String tokenizer-path (into-array String []))
            t (HuggingFaceTokenizer/newInstance path)]
        (assoc this :tokenizer t))))
  (stop [this]
    (.close this)
    (assoc this :tokenizer nil)))

(defn create
  "Build and start a Tokenizer. Use with `with-open` for one-shot use."
  [config]
  {:pre [(m/validate Config config)]}
  (start (->Tokenizer (:tokenizer-path config) nil)))

(defn component
  "Build an unstarted Tokenizer for use in a Stuart Sierra Component system."
  [config]
  {:pre [(m/validate Config config)]}
  (->Tokenizer (:tokenizer-path config) nil))

(defn close
  "Close the Tokenizer. Same as `(.close tokenizer)`."
  [^Tokenizer tokenizer]
  (.close tokenizer))

(defn encode
  "Encode `text` using the underlying HuggingFaceTokenizer.
  Returns the raw DJL Encoding; convert with get-ids / get-mask as needed."
  [{:keys [^HuggingFaceTokenizer tokenizer]} ^String text]
  (HuggingFaceTokenizer/.encode tokenizer text))

(defn get-ids
  "Extract the long[] of token ids from an Encoding."
  [^Encoding encoding]
  (Encoding/.getIds encoding))

(defn get-mask
  "Extract the long[] attention mask from an Encoding."
  [^Encoding encoding]
  (Encoding/.getAttentionMask encoding))
