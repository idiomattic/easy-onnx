(ns easy-onnx.embedder
  (:require [easy-onnx.onnx :as onnx]
            [easy-onnx.tokenizer :as tokenizer])
  (:import [smile.clustering DBSCAN]
           [smile.manifold UMAP UMAP$Options]
           [smile.math MathEx]
           [smile.math.distance Distance]))

;; ---------------------------------------------------------------------------
;; Embedding
;; ---------------------------------------------------------------------------

(defn- mean-pool
  "Mean-pool token embeddings masked by attention mask.
    hidden-state is float[1][seq_len][384], mask is long[seq_len].
    Returns a float[384]."
  [hidden-state mask]
  (let [tokens (aget ^"[[[F" hidden-state 0)
        seq-len (alength ^"[[F" tokens)
        embed-dim (alength ^"[F" (aget ^"[[F" tokens 0))
        result (float-array embed-dim)
        mask-sum (float (areduce ^longs mask i acc 0.0
                                 (+ acc (aget ^longs mask i))))]
    (dotimes [i seq-len]
      (when (pos? (aget ^longs mask i))
        (let [token-vec ^floats (aget ^"[[F" tokens i)]
          (dotimes [j embed-dim]
            (aset result j (+ (aget result j) (aget token-vec j)))))))
    (dotimes [j embed-dim]
      (aset result j (/ (aget result j) mask-sum)))
    result))

(defn embed!
  "Generate a float[384] embedding for the given text."
  [{:keys [onnx-model tokenizer]} text]
  (let [encoding (tokenizer/encode! tokenizer text)
        ids (tokenizer/get-ids encoding)
        mask (tokenizer/get-mask encoding)
        type-ids (long-array (count ids))
        inputs {"input_ids" (into-array [ids])
                "attention_mask" (into-array [mask])
                "token_type_ids" (into-array [type-ids])}
        outputs (onnx/run-model! onnx-model inputs)]
    (mean-pool (get outputs "last_hidden_state") mask)))

;; ---------------------------------------------------------------------------
;; Similarity & distance
;; ---------------------------------------------------------------------------

(defn cosine-similarity
  "Cosine similarity between two float[] embeddings. Returns a value in [-1, 1]."
  ^double [^floats a ^floats b]
  (MathEx/cosine a b))

(defn cosine-distance
  "Cosine distance (1 - similarity). Returns a value in [0, 2]."
  ^double [^floats a ^floats b]
  (- 1.0 (MathEx/cosine a b)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- floats->doubles
  "Convert a float[] to a double[]."
  ^doubles [^floats fa]
  (let [len (alength fa)
        da (double-array len)]
    (dotimes [i len]
      (aset da i (double (aget fa i))))
    da))

(def ^:private cosine-distance-fn
  "A reified smile Distance<double[]> using cosine distance."
  (reify Distance
    (d [_ a b]
      (- 1.0 (MathEx/cosine ^doubles a ^doubles b)))))

;; ---------------------------------------------------------------------------
;; Clustering
;; ---------------------------------------------------------------------------

(defn cluster
  "Run DBSCAN on a seq of float[] embeddings.
  Returns a vector of cluster indices (0-based), where -1 means unclustered/noise.
  Options:
    :radius  - neighborhood radius in cosine distance (default 0.75)
    :min-pts - minimum neighbors for a core point (default 1)"
  [embeddings & {:keys [radius min-pts]
                 :or {radius 0.75 min-pts 1}}]
  (let [data (into-array (map floats->doubles embeddings))
        result (DBSCAN/fit data cosine-distance-fn (int min-pts) (double radius))]
    (vec (.group result))))

;; ---------------------------------------------------------------------------
;; 2D projection
;; ---------------------------------------------------------------------------

(defn project-2d
  "Reduce embeddings to 2D positions via UMAP.
  Takes a seq of float[] embeddings, returns a vector of [x y] pairs."
  [embeddings]
  (let [data (into-array ^"[D" (map floats->doubles embeddings))
        options (UMAP$Options. 15)
        coords (UMAP/fit data options)]
    (mapv (fn [^doubles row]
            [(aget row 0) (aget row 1)])
          coords)))
