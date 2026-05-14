(ns easy-onnx.analysis-test
  (:require [clojure.test :refer [deftest is testing]]
            [easy-onnx.analysis :as analysis]))

(defn- v [& xs]
  (float-array xs))

(deftest cluster-groups-near-vectors
  (testing "DBSCAN groups close vectors and separates distant ones"
    (let [;; cluster A: near unit vector along x
          a1 (v 1.0 0.0 0.0)
          a2 (v 0.99 0.05 0.0)
          ;; cluster B: near unit vector along y
          b1 (v 0.0 1.0 0.0)
          b2 (v 0.05 0.99 0.0)
          groups (analysis/cluster [a1 a2 b1 b2] :radius 0.05 :min-pts 1)]
      (is (= (groups 0) (groups 1)) "a1 and a2 should share a cluster")
      (is (= (groups 2) (groups 3)) "b1 and b2 should share a cluster")
      (is (not= (groups 0) (groups 2))
          "A and B should be in different clusters"))))

(deftest project-2d-returns-correct-shape
  (testing "projection returns [x y] pairs for each input vector"
    (let [vectors [(v 1.0 0.0 0.0)
                   (v 0.0 1.0 0.0)
                   (v 0.0 0.0 1.0)
                   (v 1.0 1.0 0.0)
                   (v 0.0 1.0 1.0)]
          coords (analysis/project-2d vectors)]
      (is (= 5 (count coords)))
      (is (every? #(= 2 (count %)) coords))
      (is (every? #(every? number? %) coords)))))

(deftest cosine-similarity-of-identical-vectors
  (testing "identical vectors have similarity ~1.0"
    (let [a (v 1.0 2.0 3.0 4.0)]
      (is (< 0.999 (analysis/cosine-similarity a a))))))

(deftest cosine-similarity-of-orthogonal-vectors
  (testing "orthogonal unit vectors have similarity ~0.0"
    (let [a (v 1.0 0.0)
          b (v 0.0 1.0)]
      (is (< (Math/abs (analysis/cosine-similarity a b)) 0.001)))))

(deftest cosine-similarity-of-opposite-vectors
  (testing "opposite-direction vectors have similarity ~-1.0"
    (let [a (v 1.0 2.0 3.0)
          b (v -1.0 -2.0 -3.0)]
      (is (> -0.999 (analysis/cosine-similarity a b))))))

(deftest cosine-distance-of-identical-vectors
  (testing "identical vectors have distance ~0.0"
    (let [a (v 1.0 2.0 3.0)]
      (is (< (analysis/cosine-distance a a) 0.001)))))

(deftest cosine-distance-of-orthogonal-vectors
  (testing "orthogonal unit vectors have distance ~1.0"
    (let [a (v 1.0 0.0)
          b (v 0.0 1.0)]
      (is (< 0.999 (analysis/cosine-distance a b))))))
