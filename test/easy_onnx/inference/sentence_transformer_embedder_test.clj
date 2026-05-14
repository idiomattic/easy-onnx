(ns easy-onnx.inference.sentence-transformer-embedder-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [easy-onnx.inference.sentence-transformer-embedder :as ste]))

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
