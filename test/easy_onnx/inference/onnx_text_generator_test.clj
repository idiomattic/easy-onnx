(ns easy-onnx.inference.onnx-text-generator-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [easy-onnx.inference.onnx-text-generator :as otg])
  (:import [java.time Duration]))

(def preset :smol-lm-2)

(def generator (atom nil))

(use-fixtures :once
  (fn [f]
    (let [g (try
              (otg/create {:preset preset :max-new-tokens 8})
              (catch Exception ex
                (throw (ex-info (str "Failed to build OnnxTextGenerator. "
                                     "First run downloads ~700MB from HuggingFace to "
                                     "~/.cache/inference4j/. Ensure network access "
                                     "(or a populated cache) is available.")
                                {:preset preset}
                                ex))))]
      (reset! generator g)
      (try
        (f)
        (finally
          (.close ^java.lang.AutoCloseable g)
          (reset! generator nil))))))

(deftest generate-returns-result-map
  (testing "generate produces a map with text, token counts, and duration"
    (let [result (otg/generate @generator "Hello")]
      (is (string? (:text result)))
      (is (pos? (count (:text result))))
      (is (pos-int? (:prompt-tokens result)))
      (is (pos-int? (:generated-tokens result)))
      (is (instance? Duration (:duration result))))))

(deftest max-new-tokens-bounds-generated-tokens
  (testing "with :max-new-tokens 8, the result's :generated-tokens is <= 8"
    (with-open [g (otg/create {:preset preset :max-new-tokens 8})]
      (let [result (otg/generate g "Tell me a long story about anything")]
        (is (<= (:generated-tokens result) 8))))))

(deftest greedy-decoding-is-deterministic
  (testing "with :temperature 0.0 (greedy), two generate calls produce the same text"
    (with-open [g (otg/create {:preset preset :max-new-tokens 16 :temperature 0.0})]
      (let [a (otg/generate g "Two plus two is")
            b (otg/generate g "Two plus two is")]
        (is (= (:text a) (:text b)))))))

(deftest top-k-and-top-p-are-accepted
  (testing ":top-k and :top-p pass through without error"
    (with-open [g (otg/create {:preset preset
                               :max-new-tokens 8
                               :temperature 0.7
                               :top-k 50
                               :top-p 0.9})]
      (let [result (otg/generate g "Hi")]
        (is (pos-int? (:generated-tokens result)))))))

(deftest generate-streaming-fires-on-token-and-returns-result
  (testing "on-token is called per token; concatenated tokens equal the full :text"
    (with-open [g (otg/create {:preset preset :max-new-tokens 8 :temperature 0.0})]
      (let [tokens (atom [])
            result (otg/generate-streaming
                    g
                    "Hello"
                    (fn [token] (swap! tokens conj token)))]
        (is (pos? (count @tokens)))
        (is (= (:text result) (apply str @tokens)))))))

(deftest bogus-base-dir-throws
  (testing ":base-dir pointing at a nonexistent directory throws on create"
    (is (thrown? Exception
                 (otg/create {:preset preset
                              :base-dir "/nonexistent/base/dir"})))))

(deftest session-options-are-accepted
  (testing ":session-options accepts thread + optimization options"
    (with-open [g (otg/create {:preset preset
                               :max-new-tokens 4
                               :session-options {:intra-op-num-threads 1
                                                 :inter-op-num-threads 1
                                                 :optimization-level :all}})]
      (let [result (otg/generate g "Hi")]
        (is (pos-int? (:generated-tokens result)))))))

(deftest invalid-optimization-level-fails-precondition
  (testing "an unknown :optimization-level fails the malli precondition"
    (is (thrown? AssertionError
                 (otg/create {:preset preset
                              :session-options {:optimization-level :bogus}})))))
