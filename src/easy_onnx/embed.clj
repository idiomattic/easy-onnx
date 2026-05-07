(ns easy-onnx.embed
  "Modality-agnostic embedding helpers. No deps on other easy-onnx sections."
  (:import [smile.math MathEx]))

(defn cosine-similarity
  "Cosine similarity between two float[] vectors. Returns a value in [-1, 1]."
  ^double [^floats a ^floats b]
  (MathEx/cosine a b))

(defn cosine-distance
  "Cosine distance (1 - similarity). Returns a value in [0, 2]."
  ^double [^floats a ^floats b]
  (- 1.0 (MathEx/cosine a b)))
