(ns easy-onnx.embed.text-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [easy-onnx.embed.text :as embed-text]
            [easy-onnx.runtime :as runtime]
            [easy-onnx.tokenizer :as tokenizer]))

(def model-path "resources/ml/all-MiniLM-L6-v2/model.onnx")
(def tokenizer-path "resources/ml/all-MiniLM-L6-v2/tokenizer.json")

(defn- fixtures-present? []
  (and (.exists (java.io.File. ^String model-path))
       (.exists (java.io.File. ^String tokenizer-path))))

(def ^:dynamic *system* nil)

(use-fixtures :once
  (fn [f]
    (if (fixtures-present?)
      (with-open [r (runtime/create {:model-path model-path})
                  t (tokenizer/create {:tokenizer-path tokenizer-path})]
        (binding [*system* {:runtime r :tokenizer t}]
          (f)))
      (do
        (println "Skipping easy-onnx.embed.text-test: missing fixtures")
        (f)))))

(defn- embed [text]
  (embed-text/embed *system* text))

(deftest embedding-has-correct-dimensionality
  (testing "embedding is a 384-dim float array (MiniLM-L6-v2)"
    (when *system*
      (let [result (embed "hello world")]
        (is (instance? (Class/forName "[F") result))
        (is (= 384 (alength ^floats result)))))))

(deftest embedding-values-are-finite
  (testing "no NaN or Inf values in embedding"
    (when *system*
      (let [result (embed "a note about meetings")]
        (is (every? #(Float/isFinite %) (seq result)))))))

(deftest different-texts-produce-different-embeddings
  (testing "distinct inputs don't produce identical embeddings"
    (when *system*
      (let [e1 (embed "buy groceries")
            e2 (embed "deploy to production")]
        (is (not= (seq e1) (seq e2)))))))

(deftest short-text-still-embeds
  (testing "very short text still produces a valid embedding"
    (when *system*
      (let [result (embed "hi")]
        (is (= 384 (alength ^floats result)))
        (is (every? #(Float/isFinite %) (seq result)))))))
