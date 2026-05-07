(ns easy-onnx.analysis
  "Embedding analysis: DBSCAN clustering and UMAP 2D projection.
  Operates on raw float[] vectors; not coupled to any text-specific code."
  (:import [smile.clustering DBSCAN]
           [smile.manifold UMAP UMAP$Options]
           [smile.math MathEx]
           [smile.math.distance Distance]))

(defn- floats->doubles
  ^doubles [^floats fa]
  (let [len (alength fa)
        da (double-array len)]
    (dotimes [i len]
      (aset da i (double (aget fa i))))
    da))

(def ^:private cosine-distance-fn
  "A reified Smile Distance<double[]> using cosine distance."
  (reify Distance
    (d [_ a b]
      (- 1.0 (MathEx/cosine ^doubles a ^doubles b)))))

(defn cluster
  "Run DBSCAN on a seq of float[] embeddings.
  Returns a vector of cluster indices (0-based); -1 means unclustered/noise.
  Options:
    :radius  - neighborhood radius in cosine distance (default 0.75)
    :min-pts - minimum neighbors for a core point (default 1)"
  [embeddings & {:keys [radius min-pts] :or {radius 0.75 min-pts 1}}]
  (let [data (into-array (map floats->doubles embeddings))
        result (DBSCAN/fit data cosine-distance-fn (int min-pts) (double radius))]
    (vec (.group result))))

(defn project-2d
  "Reduce embeddings to 2D positions via UMAP.
  Takes a seq of float[] embeddings; returns a vector of [x y] pairs."
  [embeddings]
  (let [data (into-array ^"[D" (map floats->doubles embeddings))
        options (UMAP$Options. 15)
        coords (UMAP/fit data options)]
    (mapv (fn [^doubles row]
            [(aget row 0) (aget row 1)])
          coords)))
