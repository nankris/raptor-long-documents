# #074 RAPTOR vs flat chunks on QuALITY

GPU: Colab L4. Reader/summariser: Qwen/Qwen3-4B-Instruct-2507 (4B, fp16, no thinking; answer = argmax logit over A-D). Embeddings: BAAI/bge-small-en-v1.5 (33M). Data: emozilla/quality validation, 8 articles (seed 0), 145 questions.

| Arm | Accuracy | n | Mean context tokens |
|---|---|---|---|
| zero-context | 39.3% | 145 | 0 |
| flat (100-word chunks, top-k, 1500-tok budget) | 66.9% | 145 | 1471 |
| raptor (collapsed tree, 1500-tok budget) | 66.9% | 145 | 1475 |
| full article | 75.2% | 145 | 4958 |

Tree build: 245 s total, 82 summary calls (nodes by level: 299 leaves, 63, 14, 5). KMeans groups of ~5, up to 3 summary levels.

GPU time: ~21.6 min (session created ~03:39 IST, stopped 04:00:37 IST) ≈ 0.55 CU (21.6/60 x 1.54). Balance after stop: 187.28 CU. Session confirmed gone via `colab sessions`.
Caveat: 145 questions, 8 articles; 95% CI is about ±8 pts, so flat = raptor is a tie.
