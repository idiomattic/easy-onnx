(ns easy-onnx.inference.sentence-transformer-embedder-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [com.stuartsierra.component :as component]
            [easy-onnx.inference.sentence-transformer-embedder :as ste])
  (:import [io.github.inference4j.nlp SentenceTransformerEmbedder]))

(def model-id "inference4j/all-MiniLM-L6-v2")

(def embedder (atom nil))

(use-fixtures :once
  (fn [f]
    (let [e (try
              (ste/create {:model-id model-id})
              (catch Exception ex
                (throw (ex-info (str "Failed to build SentenceTransformerEmbedder. "
                                     "First run downloads ~80MB from HuggingFace to ~/.cache/inference4j/. "
                                     "Ensure network access (or a populated cache) is available.")
                                {:model-id model-id}
                                ex))))]
      (reset! embedder e)
      (try
        (f)
        (finally
          (.close ^java.lang.AutoCloseable e)
          (reset! embedder nil))))))

(deftest encode-returns-384-float-array
  (testing "encode produces a float[] of MiniLM's hidden size (384)"
    (let [result (ste/encode @embedder "hello world")]
      (is (instance? (Class/forName "[F") result))
      (is (= 384 (alength ^floats result))))))

(deftest encode-returns-finite-values
  (testing "no NaN or Inf values in embedding"
    (let [result (ste/encode @embedder "a note about meetings")]
      (is (every? #(Float/isFinite %) (seq result))))))

(deftest different-texts-produce-different-embeddings
  (testing "distinct inputs produce distinct embeddings"
    (let [e1 (ste/encode @embedder "buy groceries")
          e2 (ste/encode @embedder "deploy to production")]
      (is (not= (seq e1) (seq e2))))))

(deftest encode-batch-returns-one-embedding-per-input
  (testing "encode-batch preserves input order and count"
    (let [texts ["foo" "bar baz" "the quick brown fox"]
          results (ste/encode-batch @embedder texts)]
      (is (= 3 (count results)))
      (is (every? #(instance? (Class/forName "[F") %) results))
      (is (every? #(= 384 (alength ^floats %)) results)))))

(deftest encode-batch-matches-individual-encodes
  (testing "encode-batch results equal individual encode results"
    (let [texts ["one" "two"]
          batch (ste/encode-batch @embedder texts)
          singles (mapv #(ste/encode @embedder %) texts)]
      (is (= (mapv seq batch) (mapv seq singles))))))

(deftest normalize-produces-unit-norm-vectors
  (testing "with :normalize? true, embeddings have L2 norm ~1.0"
    (with-open [e (ste/create {:model-id model-id :normalize? true})]
      (let [v (ste/encode e "normalize me")
            norm (Math/sqrt (areduce ^floats v i acc 0.0
                                     (+ acc (* (aget ^floats v i)
                                               (aget ^floats v i)))))]
        (is (< (Math/abs (- 1.0 norm)) 0.001))))))

(deftest pooling-cls-differs-from-mean
  (testing "CLS pooling and MEAN pooling produce different embeddings for the same input"
    (with-open [mean-e (ste/create {:model-id model-id :pooling :mean})
                cls-e  (ste/create {:model-id model-id :pooling :cls})]
      (let [text "this sentence is for testing pooling"
            mean-v (ste/encode mean-e text)
            cls-v  (ste/encode cls-e text)]
        (is (not= (seq mean-v) (seq cls-v)))))))

(deftest text-prefix-changes-embedding
  (testing "applying a text-prefix changes the resulting embedding"
    (with-open [plain    (ste/create {:model-id model-id})
                prefixed (ste/create {:model-id model-id :text-prefix "query: "})]
      (let [text "what is the capital of France"
            v1 (ste/encode plain text)
            v2 (ste/encode prefixed text)]
        (is (not= (seq v1) (seq v2)))))))

(deftest max-length-truncates-long-input
  (testing ":max-length truncates input to keep encode fast and bounded"
    (with-open [e (ste/create {:model-id model-id :max-length 16})]
      (let [long-text (apply str (repeat 200 "the quick brown fox "))
            result (ste/encode e long-text)]
        (is (= 384 (alength ^floats result)))
        (is (every? #(Float/isFinite %) (seq result)))))))

(deftest base-dir-loads-from-local-cache
  (testing ":base-dir + :model-id resolve to <base-dir>/<model-id> via LocalModelSource"
    (let [cache-home (System/getProperty "user.home")
          base-dir (str cache-home "/.cache/inference4j")
          sub-id "inference4j/all-MiniLM-L6-v2"]
      ;; Precondition: the model was already cached when @embedder was built.
      (with-open [e (ste/create {:base-dir base-dir :model-id sub-id})]
        (let [v (ste/encode e "local source test")]
          (is (= 384 (alength ^floats v))))))))

(deftest bogus-base-dir-throws
  (testing ":base-dir pointing at a nonexistent directory throws on create, even when :model-id is otherwise valid on HuggingFace"
    (is (thrown? Exception
                 (ste/create {:base-dir "/nonexistent/base/dir"
                              :model-id model-id})))))

(deftest session-options-are-accepted
  (testing ":session-options accepts thread + optimization options without failure"
    (with-open [e (ste/create {:model-id model-id
                               :session-options {:intra-op-num-threads 1
                                                 :inter-op-num-threads 1
                                                 :optimization-level :all}})]
      (let [v (ste/encode e "session options test")]
        (is (= 384 (alength ^floats v)))))))

(deftest invalid-optimization-level-fails
  (testing "an unknown :optimization-level fails the malli precondition"
    (is (thrown? AssertionError
                 (ste/create {:model-id model-id
                              :session-options {:optimization-level :bogus}})))))

(deftest component-returns-unstarted
  (testing "component returns an Embedder with no underlying SentenceTransformerEmbedder"
    (let [c (ste/component {:model-id model-id})]
      (is (nil? (:embedder c))))))

(deftest component-start-populates-and-stop-clears
  (testing "component start builds the underlying embedder; stop closes and clears it"
    (let [c (ste/component {:model-id model-id})
          started (component/start c)]
      (is (instance? SentenceTransformerEmbedder (:embedder started)))
      (let [stopped (component/stop started)]
        (is (nil? (:embedder stopped)))))))

(deftest missing-model-id-fails-precondition
  (testing "create without :model-id fails the malli precondition"
    (is (thrown? AssertionError (ste/create {})))))

(deftest invalid-pooling-fails-precondition
  (testing "an unknown :pooling value fails the malli precondition"
    (is (thrown? AssertionError
                 (ste/create {:model-id model-id :pooling :weighted})))))

(deftest nonexistent-model-id-throws
  (testing "a model-id that doesn't exist on HuggingFace throws on create"
    (is (thrown? Exception
                 (ste/create {:model-id "inference4j/this-model-does-not-exist"})))))
