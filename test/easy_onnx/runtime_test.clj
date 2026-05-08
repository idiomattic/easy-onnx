(ns easy-onnx.runtime-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [com.stuartsierra.component :as component]
            [easy-onnx.runtime :as runtime])
  (:import [ai.onnxruntime OrtEnvironment OrtSession]))

(def model-path "resources/ml/all-MiniLM-L6-v2/model.onnx")

(defn- fixtures-present? []
  (.exists (java.io.File. ^String model-path)))

(def fixtures? (atom false))

(use-fixtures :each (fn [f]
                      (reset! fixtures? (fixtures-present?))
                      (when-not @fixtures?
                        (throw (ex-info "Skipping easy-onnx.runtime-test: missing fixture at" {:model-path model-path})))
                      (f)))

(deftest create-returns-started-session
  (testing "create returns a Session with env and session populated"
    (with-open [s (runtime/create {:model-path model-path})]
      (is (instance? OrtEnvironment (:env s)))
      (is (instance? OrtSession (:session s))))))

(deftest run-model-returns-output-map
  (testing "run-model returns a map keyed by output name"
    (with-open [s (runtime/create {:model-path model-path})]
      (let [inputs {"input_ids" (into-array [(long-array [101 7592 102])])
                    "attention_mask" (into-array [(long-array [1 1 1])])
                    "token_type_ids" (into-array [(long-array [0 0 0])])}
            outputs (runtime/run-model s inputs)]
        (is (contains? outputs "last_hidden_state"))))))

(deftest with-open-closes-session
  (testing "after with-open exits, the underlying OrtSession is closed"
    (let [s (runtime/create {:model-path model-path})
          raw (:session s)]
      (.close s)
      (is (thrown? Exception
                   (runtime/run-model
                    {:env (:env s) :session raw}
                    {"input_ids" (into-array [(long-array [101])])
                     "attention_mask" (into-array [(long-array [1])])
                     "token_type_ids" (into-array [(long-array [0])])}))))))

(deftest component-lifecycle
  (testing "component returns an unstarted Session that start populates"
    (let [c (runtime/component {:model-path model-path})]
      (is (nil? (:session c)) "unstarted has no session")
      (let [started (component/start c)]
        (is (instance? OrtEnvironment (:env started)))
        (is (instance? OrtSession (:session started)))
        (let [stopped (component/stop started)]
          (is (nil? (:session stopped)) "stopped has no session"))))))

(deftest init-exception-propagates
  (testing "create with a nonexistent model path throws"
    (is (thrown? Exception
                 (runtime/create {:model-path "/nonexistent/model.onnx"})))))

(deftest invalid-config-fails-precondition
  (testing "missing :model-path fails the malli precondition"
    (is (thrown? AssertionError
                 (runtime/create {})))))

(deftest session-options-are-accepted
  (testing "create accepts thread / optimization / log-level options and runs"
    (with-open [s (runtime/create {:model-path model-path
                                   :intra-op-num-threads 1
                                   :inter-op-num-threads 1
                                   :optimization-level :all
                                   :log-level :warning})]
      (is (instance? OrtSession (:session s)))
      (let [inputs {"input_ids" (into-array [(long-array [101 7592 102])])
                    "attention_mask" (into-array [(long-array [1 1 1])])
                    "token_type_ids" (into-array [(long-array [0 0 0])])}
            outputs (runtime/run-model s inputs)]
        (is (contains? outputs "last_hidden_state"))))))

(deftest invalid-optimization-level-fails
  (testing "an unknown :optimization-level fails the malli precondition"
    (is (thrown? AssertionError
                 (runtime/create {:model-path model-path
                                  :optimization-level :bogus})))))

(deftest invalid-log-level-fails
  (testing "an unknown :log-level fails the malli precondition"
    (is (thrown? AssertionError
                 (runtime/create {:model-path model-path
                                  :log-level :loud})))))
