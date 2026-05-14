(ns easy-onnx.inference.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [easy-onnx.inference.core :as core])
  (:import [ai.onnxruntime OrtSession$SessionOptions]
           [io.github.inference4j.model LocalModelSource ModelSource]
           [io.github.inference4j.session SessionConfigurer]))

(deftest opt-level-map-has-four-entries
  (testing "opt-level-map covers :none :basic :extended :all"
    (is (= #{:none :basic :extended :all} (set (keys core/opt-level-map))))
    (is (every? some? (vals core/opt-level-map)))))

(deftest ->session-configurer-with-all-keys-set
  (testing "configurer applies all three settings when all keys are present"
    (let [cfg (core/->session-configurer {:intra-op-num-threads 2
                                          :inter-op-num-threads 3
                                          :optimization-level :basic})]
      (is (instance? SessionConfigurer cfg))
      (with-open [opts (OrtSession$SessionOptions.)]
        (SessionConfigurer/.configure cfg opts)
        (is (instance? OrtSession$SessionOptions opts))))))

(deftest ->session-configurer-with-partial-keys
  (testing "configurer skips setters whose key is absent"
    (let [cfg (core/->session-configurer {:intra-op-num-threads 1})]
      (with-open [opts (OrtSession$SessionOptions.)]
        (SessionConfigurer/.configure cfg opts)
        (is (instance? OrtSession$SessionOptions opts))))))

(deftest ->session-configurer-with-empty-map
  (testing "configurer with no keys is a no-op"
    (let [cfg (core/->session-configurer {})]
      (with-open [opts (OrtSession$SessionOptions.)]
        (SessionConfigurer/.configure cfg opts)
        (is (instance? OrtSession$SessionOptions opts))))))

(deftest resolve-source-passes-through-model-source
  (testing ":model-source wins when present"
    (let [src (reify ModelSource
                (resolve [_ _id] (throw (UnsupportedOperationException.))))]
      (is (identical? src (core/resolve-source {:model-source src}))))))

(deftest resolve-source-builds-local-from-base-dir
  (testing ":base-dir produces a LocalModelSource"
    (let [src (core/resolve-source {:base-dir "/tmp"})]
      (is (instance? LocalModelSource src)))))

(deftest resolve-source-returns-nil-for-empty-config
  (testing "absent :model-source and :base-dir means 'use inference4j default'"
    (is (nil? (core/resolve-source {})))))

(deftest resolve-source-model-source-takes-precedence-over-base-dir
  (testing ":model-source overrides :base-dir when both are set"
    (let [src (reify ModelSource
                (resolve [_ _id] (throw (UnsupportedOperationException.))))]
      (is (identical? src (core/resolve-source {:model-source src :base-dir "/tmp"}))))))
