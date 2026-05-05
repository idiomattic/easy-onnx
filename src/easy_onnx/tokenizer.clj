(ns easy-onnx.tokenizer
  (:require [malli.core :as m]
            [mokujin.log :as log]
            [utility-belt.component :as util.component])
  (:import [ai.djl.huggingface.tokenizers Encoding HuggingFaceTokenizer]
           [java.nio.file Path]))

(def TokenizerConfig
  (m/schema
   [:map
    [:tokenizer-path [:string {:min 1}]]]))

(defn create
  [{:keys [tokenizer-path] :as config}]
  {:pre [(m/validate TokenizerConfig config)]}
  (util.component/map->component
   {:init {:tokenizer-path tokenizer-path}
    :start (fn [this]
             (if (:tokenizer this)
               this
               (try
                 (let [path (Path/of ^String tokenizer-path (into-array String []))
                       tokenizer (HuggingFaceTokenizer/newInstance path)]
                   (assoc this :tokenizer tokenizer))
                 (catch Throwable e
                   (log/error e "could not initialize tokenizer" {:tokenizer-path tokenizer-path})))))
    :stop (fn [this]
            (when-let [^HuggingFaceTokenizer t (:tokenizer this)]
              (HuggingFaceTokenizer/.close t))
            (assoc this :tokenizer nil))}))

(defn encode!
  [^HuggingFaceTokenizer tokenizer ^String text]
  (HuggingFaceTokenizer/.encode (:tokenizer tokenizer) text))

(defn get-ids [encoding]
  (Encoding/.getIds ^Encoding encoding))

(defn get-mask [encoding]
  (Encoding/.getAttentionMask ^Encoding encoding))

