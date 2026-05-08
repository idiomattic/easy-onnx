(ns easy-onnx.tokenizer
  (:require [com.stuartsierra.component :as component]
            [malli.core :as m])
  (:import [ai.djl.huggingface.tokenizers Encoding HuggingFaceTokenizer]
           [java.lang AutoCloseable]
           [java.nio.file Path]))

;; NOTE: Reconsider malli-vs-plain-validation if config schemas stay trivial
;; after the API stabilizes. See spec, Future considerations.
(def Config
  (m/schema
   [:map
    [:tokenizer-path [:string {:min 1}]]
    [:truncation? {:optional true} :boolean]
    [:max-length {:optional true} [:int {:min 1}]]
    [:padding? {:optional true} :boolean]
    [:pad-to-max-length? {:optional true} :boolean]
    [:pad-to-multiple-of {:optional true} [:int {:min 1}]]
    [:add-special-tokens? {:optional true} :boolean]]))

(defn- build-tokenizer
  "Build a HuggingFaceTokenizer from a validated config map."
  ^HuggingFaceTokenizer
  [{:keys [tokenizer-path
           truncation?
           max-length
           padding?
           pad-to-max-length?
           pad-to-multiple-of
           add-special-tokens?]}]
  (let [path (Path/of ^String tokenizer-path (into-array String []))
        builder (-> (HuggingFaceTokenizer/builder)
                    (.optTokenizerPath path))]
    (-> builder
        (cond->
         (some? truncation?) (.optTruncation truncation?)
         (some? max-length) (.optMaxLength max-length)
         (some? padding?) (.optPadding padding?)
         pad-to-max-length? (.optPadToMaxLength)
         (some? pad-to-multiple-of) (.optPadToMultipleOf pad-to-multiple-of)
         (some? add-special-tokens?) (.optAddSpecialTokens add-special-tokens?))
        (.build))))

(defrecord Tokenizer [;; Configuration
                      tokenizer-path
                      truncation?
                      max-length
                      padding?
                      pad-to-max-length?
                      pad-to-multiple-of
                      add-special-tokens?

                      ;; Managed
                      tokenizer]
  AutoCloseable
  (close [_]
    (when tokenizer
      (HuggingFaceTokenizer/.close ^HuggingFaceTokenizer tokenizer)))

  component/Lifecycle
  (start [this]
    (if tokenizer
      this
      (assoc this :tokenizer (build-tokenizer this))))
  (stop [this]
    (.close this)
    (assoc this :tokenizer nil)))

(defn create
  "Build and start a Tokenizer. Use with `with-open` for one-shot use.

  Required:
    :tokenizer-path - Path to a HuggingFace tokenizer.json file.

  Optional builder settings (passed through to DJL's HuggingFaceTokenizer):
    :truncation?         - boolean. Enable truncation to :max-length. Note:
                           DJL truncates at :max-length even without this set;
                           set explicitly to be safe across DJL versions.
    :max-length          - int. Maximum sequence length.
    :padding?            - boolean. Pad to the longest input in a batch
                           (no-op for single-text encoding).
    :pad-to-max-length?  - boolean. Pad encoded sequences out to :max-length.
                           Useful for fixed-shape tensor inputs.
    :pad-to-multiple-of  - int. Pad sequence length up to a multiple of this.
    :add-special-tokens? - boolean. Wrap with [CLS]/[SEP] (default per model)."
  [config]
  {:pre [(m/validate Config config)]}
  (component/start (map->Tokenizer (assoc config :tokenizer nil))))

(defn component
  "Build an unstarted Tokenizer for use in a Stuart Sierra Component system.
  See `create` for accepted config keys."
  [config]
  {:pre [(m/validate Config config)]}
  (map->Tokenizer (assoc config :tokenizer nil)))

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
