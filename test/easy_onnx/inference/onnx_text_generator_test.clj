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
