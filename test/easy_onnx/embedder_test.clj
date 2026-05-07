(ns easy-onnx.embedder-test
  (:require [clojure.test :refer [deftest is testing]]
            [com.stuartsierra.component :as component]
            [easy-onnx.embedder :as embedder]
            [easy-onnx.onnx :as onnx]
            [easy-onnx.tokenizer :as tokenizer]))

(defn onnx-model []
  (-> (onnx/create {:model-path "resources/ml/all-MiniLM-L6-v2/model.onnx"})
      (component/start)))

(defn tokenizer []
  (-> (tokenizer/create {:tokenizer-path "resources/ml/all-MiniLM-L6-v2/tokenizer.json"})
      (component/start)))

(defn test-system []
  {:onnx-model (onnx-model)
   :tokenizer (tokenizer)})

(defn- embed! [text]
  (embedder/embed! (test-system) text))

;; ---------------------------------------------------------------------------
;; Embedding
;; ---------------------------------------------------------------------------

(deftest embedding-has-correct-dimensionality
  (testing "embedding is a 384-dimensional float array"
    (let [result (embed! "hello world")]
      (is (instance? (Class/forName "[F") result))
      (is (= 384 (alength ^floats result))))))

(deftest embedding-values-are-finite
  (testing "no NaN or Inf values in embedding"
    (let [result (embed! "a note about meetings")]
      (is (every? #(Float/isFinite %) (seq result))))))

(deftest different-texts-produce-different-embeddings
  (testing "distinct inputs don't produce identical embeddings"
    (let [e1 (embed! "buy groceries")
          e2 (embed! "deploy to production")]
      (is (not= (seq e1) (seq e2))))))

(deftest short-text-still-embeds
  (testing "very short text produces a valid embedding"
    (let [result (embed! "hi")]
      (is (= 384 (alength ^floats result)))
      (is (every? #(Float/isFinite %) (seq result))))))

;; ---------------------------------------------------------------------------
;; Similarity & distance
;; ---------------------------------------------------------------------------

(deftest cosine-similarity-of-identical-embeddings
  (testing "identical embeddings have similarity ~1.0"
    (let [e (embed! "hello world")]
      (is (< 0.999 (embedder/cosine-similarity e e))))))

(deftest cosine-distance-of-identical-embeddings
  (testing "identical embeddings have distance ~0.0"
    (let [e (embed! "hello world")]
      (is (< (embedder/cosine-distance e e) 0.001)))))

(deftest similar-texts-are-closer-than-dissimilar
  (testing "semantically similar texts have higher cosine similarity"
    (let [e1 (embed! "meeting with the team about project deadlines")
          e2 (embed! "team standup to discuss upcoming milestones")
          e3 (embed! "recipe for chocolate cake")]
      (is (> (embedder/cosine-similarity e1 e2)
             (embedder/cosine-similarity e1 e3))))))

;; ---------------------------------------------------------------------------
;; Clustering
;; ---------------------------------------------------------------------------

(deftest cluster-groups-similar-embeddings
  (testing "related notes cluster together, unrelated ones separate"
    (let [work-1 (embed! "sprint planning for Q3 deliverables")
          work-2 (embed! "backlog grooming and ticket prioritization")
          food-1 (embed! "best pizza dough recipe with sourdough starter")
          food-2 (embed! "homemade pasta with fresh basil and tomato")
          groups (embedder/cluster [work-1 work-2 food-1 food-2])]
      ;; work notes should share a cluster, food notes should share a cluster
      (is (= (groups 0) (groups 1))
          "work notes should be in the same cluster")
      (is (= (groups 2) (groups 3))
          "food notes should be in the same cluster")
      (is (not= (groups 0) (groups 2))
          "work and food should be in different clusters"))))

;; ---------------------------------------------------------------------------
;; 2D projection
;; ---------------------------------------------------------------------------

(deftest project-2d-returns-correct-shape
  (testing "projection returns [x y] pairs for each embedding"
    (let [embeddings (mapv embed! ["one" "two" "three" "four" "five"])
          coords (embedder/project-2d embeddings)]
      (is (= 5 (count coords)))
      (is (every? #(= 2 (count %)) coords))
      (is (every? #(every? number? %) coords)))))
