(ns easy-onnx.tokenizer-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [com.stuartsierra.component :as component]
            [easy-onnx.tokenizer :as tokenizer])
  (:import [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]))

(def tokenizer-path "resources/ml/all-MiniLM-L6-v2/tokenizer.json")

(defn- fixtures-present? []
  (.exists (java.io.File. ^String tokenizer-path)))

(def fixtures? (atom false))

(use-fixtures :each (fn [f]
                      (reset! fixtures? (fixtures-present?))
                      (when-not @fixtures?
                        (throw (ex-info "Skipping easy-onnx.tokenizer-test: missing fixture at" {:tokenizer-path tokenizer-path})))
                      (f)))

(deftest create-returns-started-tokenizer
  (testing "create returns a Tokenizer with the underlying tokenizer set"
    (with-open [t (tokenizer/create {:tokenizer-path tokenizer-path})]
      (is (instance? HuggingFaceTokenizer (:tokenizer t))))))

(deftest encode-returns-encoding-with-ids-and-mask
  (testing "encode returns an Encoding with non-empty ids and mask"
    (with-open [t (tokenizer/create {:tokenizer-path tokenizer-path})]
      (let [enc (tokenizer/encode t "hello world")
            ids (tokenizer/get-ids enc)
            mask (tokenizer/get-mask enc)]
        (is (pos? (alength ^longs ids)))
        (is (= (alength ^longs ids) (alength ^longs mask)))))))

(deftest component-lifecycle
  (testing "component returns an unstarted Tokenizer that start populates"
    (let [c (tokenizer/component {:tokenizer-path tokenizer-path})]
      (is (nil? (:tokenizer c)) "unstarted has no tokenizer")
      (let [started (component/start c)]
        (is (instance? HuggingFaceTokenizer (:tokenizer started)))
        (let [stopped (component/stop started)]
          (is (nil? (:tokenizer stopped)) "stopped has no tokenizer"))))))

(deftest init-exception-propagates
  (testing "create with a nonexistent tokenizer path throws"
    (is (thrown? Exception
                 (tokenizer/create {:tokenizer-path "/nonexistent/tokenizer.json"})))))

(deftest invalid-config-fails-precondition
  (testing "missing :tokenizer-path fails the malli precondition"
    (is (thrown? AssertionError
                 (tokenizer/create {})))))

(deftest truncation-caps-ids-at-max-length
  (testing "with :truncation? true and :max-length n, ids are capped at n"
    (let [long-text (apply str (repeat 200 "the quick brown fox "))]
      (with-open [t (tokenizer/create {:tokenizer-path tokenizer-path
                                       :truncation? true
                                       :max-length 32})]
        (let [ids (tokenizer/get-ids (tokenizer/encode t long-text))]
          (is (<= (alength ^longs ids) 32)))))))

(deftest pad-to-max-length-pads-short-ids
  (testing "with :pad-to-max-length? true and :max-length n, short input produces n-length ids"
    (with-open [t (tokenizer/create {:tokenizer-path tokenizer-path
                                     :pad-to-max-length? true
                                     :max-length 20})]
      (let [ids (tokenizer/get-ids (tokenizer/encode t "hi"))]
        (is (= 20 (alength ^longs ids)))))))

(deftest no-padding-leaves-natural-length
  (testing "without :pad-to-max-length?, short input keeps its natural length"
    (with-open [t (tokenizer/create {:tokenizer-path tokenizer-path})]
      (let [ids (tokenizer/get-ids (tokenizer/encode t "hi"))]
        (is (< (alength ^longs ids) 10) "expected short ids, no padding")))))
