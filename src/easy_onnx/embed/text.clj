(ns easy-onnx.embed.text
  "Text-specific embedding helpers. Currently sentence-transformer-shaped
  (input names: input_ids, attention_mask, token_type_ids; output:
  last_hidden_state, mean-pooled against the attention mask)."
  (:require [easy-onnx.runtime :as runtime]
            [easy-onnx.tokenizer :as tokenizer]))

(defn- mean-pool
  "Mean-pool token embeddings masked by attention mask.
  hidden-state is float[1][seq_len][embed_dim]; mask is long[seq_len].
  Returns a float[embed_dim]."
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

(defn embed
  "Generate an embedding (float[]) for `text` via mean-pooling.
  `deps` is a map of {:runtime <Session>, :tokenizer <Tokenizer>}."
  [{:keys [runtime tokenizer]} text]
  (let [encoding (tokenizer/encode tokenizer text)
        ids (tokenizer/get-ids encoding)
        mask (tokenizer/get-mask encoding)
        type-ids (long-array (count ids))
        inputs {"input_ids" (into-array [ids])
                "attention_mask" (into-array [mask])
                "token_type_ids" (into-array [type-ids])}
        outputs (runtime/run-model runtime inputs)]
    (mean-pool (get outputs "last_hidden_state") mask)))
