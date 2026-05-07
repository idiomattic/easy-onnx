(ns easy-onnx.embed-test
  (:require [clojure.test :refer [deftest is testing]]
            [easy-onnx.embed :as embed]))

(deftest cosine-similarity-of-identical-vectors
  (testing "identical vectors have similarity ~1.0"
    (let [v (float-array [1.0 2.0 3.0 4.0])]
      (is (< 0.999 (embed/cosine-similarity v v))))))

(deftest cosine-similarity-of-orthogonal-vectors
  (testing "orthogonal unit vectors have similarity ~0.0"
    (let [a (float-array [1.0 0.0])
          b (float-array [0.0 1.0])]
      (is (< (Math/abs (embed/cosine-similarity a b)) 0.001)))))

(deftest cosine-similarity-of-opposite-vectors
  (testing "opposite-direction vectors have similarity ~-1.0"
    (let [a (float-array [1.0 2.0 3.0])
          b (float-array [-1.0 -2.0 -3.0])]
      (is (> -0.999 (embed/cosine-similarity a b))))))

(deftest cosine-distance-of-identical-vectors
  (testing "identical vectors have distance ~0.0"
    (let [v (float-array [1.0 2.0 3.0])]
      (is (< (embed/cosine-distance v v) 0.001)))))

(deftest cosine-distance-of-orthogonal-vectors
  (testing "orthogonal unit vectors have distance ~1.0"
    (let [a (float-array [1.0 0.0])
          b (float-array [0.0 1.0])]
      (is (< 0.999 (embed/cosine-distance a b))))))
