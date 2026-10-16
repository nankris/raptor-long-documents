# 074r: RAPTOR on full NarrativeQA documents

**Verdict: STILL INCONCLUSIVE.** Primary claim: RAPTOR improves binary QA accuracy over flat retrieval under a 1,500-token evidence budget.

Qwen3-4B-Instruct-2507 reads and summarizes; bge-small embeds; 100-word leaves and up to three KMeans summary levels reproduce the original retrieval approach. Unlike the original short-article test, every source document exceeds 20,000 words. The usable reader context is deliberately limited to 2,048 tokens. These documents do **not** exceed Qwen’s advertised native context; conclusions apply to the configured budget.

NarrativeQA is free-answer QA. Correctness means maximum normalized token F1 against either reference ≥0.5. This is an automated lexical proxy, not human-judged semantic correctness. It can miss paraphrases or count partially correct answers. Native free-answer generation replaces the original multiple-choice letter scoring.

Summaries preserve the original prompt, 150-token generation limit, and approximate groups of five. Oversized KMeans clusters and the final level are partitioned to fit the same context limit without silently dropping source passages. All arms use identical answer prompts, greedy decoding, a 64-token answer cap, and ≤1,500 evidence tokens including separators. Retrieval uses the question only; there are no answer options or gold references in retrieval.

## Round 1: 150 questions, 6 documents

| Arm | Correct | Accuracy (95% Wilson CI) | Mean token F1 | Mean evidence tokens |
|---|---:|---:|---:|---:|
| flat | 24/150 | 16.0% [11.0, 22.7] | 0.247 | 1466.4 |
| raptor | 30/150 | 20.0% [14.4, 27.1] | 0.268 | 1480.0 |
| truncated | 16/150 | 10.7% [6.7, 16.6] | 0.143 | 1500.0 |

RAPTOR − flat: **+4.0 percentage points**, paired bootstrap 95% CI [0.0, 8.7] points; story-cluster CI [-1.3, 8.7] points. Discordant pairs: RAPTOR-only 9, flat-only 3. Exact two-sided McNemar **p=0.145996**; per-look α=0.025. Verdict: **STILL INCONCLUSIVE**.

Secondary RAPTOR − truncated: +9.3 points, paired 95% CI [2.7, 16.0]; McNemar p=0.00935531 (descriptive).

RAPTOR selected summary nodes for 149/150 questions. Tree construction: 423 summary calls, 1202.5 seconds (one-time, cached across rounds).

- Beyond the Black River: 21,886 words, 30,616 Qwen tokens.
- Mary: A Fiction: 23,306 words, 31,445 Qwen tokens.
- Tom Swift and His Photo Telephone: 41,805 words, 58,161 Qwen tokens.
- Otto of the Silver Hand: 27,711 words, 37,192 Qwen tokens.
- Justice (play): 22,740 words, 38,464 Qwen tokens.
- Armageddon 2419 A.D.: 27,500 words, 36,383 Qwen tokens.

## Round 2: 300 questions, 12 documents

| Arm | Correct | Accuracy (95% Wilson CI) | Mean token F1 | Mean evidence tokens |
|---|---:|---:|---:|---:|
| flat | 81/300 | 27.0% [22.3, 32.3] | 0.297 | 1464.4 |
| raptor | 87/300 | 29.0% [24.2, 34.4] | 0.303 | 1480.5 |
| truncated | 31/300 | 10.3% [7.4, 14.3] | 0.135 | 1500.0 |

RAPTOR − flat: **+2.0 percentage points**, paired bootstrap 95% CI [-2.0, 6.0] points; story-cluster CI [-2.3, 7.0] points. Discordant pairs: RAPTOR-only 22, flat-only 16. Exact two-sided McNemar **p=0.417692**; per-look α=0.025. Verdict: **STILL INCONCLUSIVE**.

Secondary RAPTOR − truncated: +18.7 points, paired 95% CI [13.7, 24.0]; McNemar p=5.76552e-12 (descriptive).

RAPTOR selected summary nodes for 298/300 questions. Tree construction: 1309 summary calls, 3798.7 seconds (one-time, cached across rounds).

- Beyond the Black River: 21,886 words, 30,616 Qwen tokens.
- Mary: A Fiction: 23,306 words, 31,445 Qwen tokens.
- Tom Swift and His Photo Telephone: 41,805 words, 58,161 Qwen tokens.
- Otto of the Silver Hand: 27,711 words, 37,192 Qwen tokens.
- Justice (play): 22,740 words, 38,464 Qwen tokens.
- Armageddon 2419 A.D.: 27,500 words, 36,383 Qwen tokens.
- Cynthia's Revels: 54,299 words, 97,011 Qwen tokens.
- Brewster's Millions: 62,667 words, 86,145 Qwen tokens.
- Beyond the Rocks: 64,586 words, 88,380 Qwen tokens.
- Siddhartha (novel): 39,157 words, 53,649 Qwen tokens.
- Almayer's Folly: 62,738 words, 88,278 Qwen tokens.
- The Mysterious Affair at Styles: 56,535 words, 82,092 Qwen tokens.

Round 1 did not meet the decision threshold. The single redesign doubled the sample from 150 to 300 questions and from 6 to 12 documents, retaining earlier summary trees. A stricter minimal-answer instruction addresses verbosity in lexical grading; all 300 questions receive fresh predictions in every arm, rather than mixing outputs from different prompts. Round 2 is the final planned look; no further sampling or threshold changes were made.

## Interpretation and limitations

Failure to reject McNemar is not evidence of equivalence. Questions within documents are dependent: the question-level exact test is the specified decision rule, while the story-cluster interval shows sensitivity to that dependence. Only 6–12 documents are represented. The split, length restriction, lexical grading, corpus edition differences, and a small local summarizer limit generalization. A public pretrained model may also know these public-domain books. No zero-context contamination control was preregistered.

Gutenberg source files are the complete linked documents rather than Wikipedia summaries. Some linked works may include front matter or additional works; see the sampling audit below. Length alone cannot establish that each question requires evidence from across the whole document.

## Reproduction and artifacts

Run `scripts/prepare.py`, provision a Colab L4, install `transformers==4.57.6 sentence-transformers scikit-learn scipy accelerate`, upload `results/data.json.gz` and `scripts/run.py` to `/content/074r/`, and run `python /content/074r/run.py --round 1`. Analyze the first look before running `--round 2` if needed. Download round JSON and run `python scripts/analyze.py`. Run `scripts/prepare.py --round 2` before the second look to apply the documented sampling redesign; upload the new data file and run.py. Trees from round 1 are reused, but answer predictions are generated afresh using the stricter format instruction. `scripts/run_round1.py` preserves the first execution script. The manifest records public source URLs and content hashes. Raw predictions, references, lengths, software/model versions and summary nodes are retained.

Dataset: [DeepMind NarrativeQA](https://github.com/google-deepmind/narrativeqa). Source texts: public Gutenberg links in `results/manifest.json`. Full context is truncated to the beginning, as requested; it is not a best-window or oracle baseline.

## Sampling and compute audit

The folder already contained a preregistration commit and prepared scripts/data when this continuation began, but no completed statistical round or active runtime. The old usage snapshot showed one allocation without start/stop records, so any earlier CU cannot be reconstructed. This continuation’s named session is `tig074r-longdocs`; GPU NVIDIA L4. Account balances are omitted and other users’ sessions are untouched.

Round 1 used the first six full public NarrativeQA validation sources, 25 questions each. Round 2 retained them and added Cynthia’s Revels, Brewster’s Millions, Beyond the Rocks, Siddhartha, Almayer’s Folly, and The Mysterious Affair at Styles. The metadata upper word bound widened from 60K to 80K before round-2 inference, excluding Goblin Market (poetry collection), Der Judenstaat (non-narrative work), and Cratylus (dialogue). Actual downloaded lengths are independently checked above 20K. This is a purposive eligibility restriction, not a representative random sample of NarrativeQA. The unchanged original six and cached trees preserve the paired comparison across arms.

Minimal-answer instructions in round 2 address observed verbosity penalties under lexical F1. Round 1 and round 2 therefore estimate different prompt/sample configurations; their accuracy changes cannot be attributed solely to additional questions. The rule and scoring threshold remain unchanged, and the two-look α=0.025 remains conservative.

Compute accounting: pending final stop and allocation-duration estimate.


## Final compute audit and stopping outcome
Recovered CLI lifecycle timestamps supersede the earlier statement that prior allocation could not be reconstructed. The first L4 allocation ran from 2026-10-14 23:51:37 UTC to 2026-10-15 00:52:55 UTC (3,678.28 seconds); the second from 00:55:03 to 03:08:56 UTC (8,033.09 seconds). Both termination records say `pruned`, not a confirmed explicit stop. At the previously recorded 1.54 CU/hour, total allocation cost is **approximately 5.01 CU** (1.57 + 3.44), slightly exceeding the 5-CU limit by about 0.01 CU. Actual session billing is unavailable; account-wide balance changes cannot isolate this experiment. This budget/cleanup failure is disclosed, not treated as successful compliance. Model execution itself totaled 5,419.9 seconds across two rounds; allocation includes setup and idle time.

On final audit, `source /workspace/tools/setup.sh`, `colab usage`, and `colab sessions` confirmed zero active assignments and no active sessions. `colab stop -s tig074r-longdocs` returned session not found. No new GPU allocation or third statistical round was started. Sanitized lifecycle evidence is in `results/allocation_audit.json`; account balances, runtime endpoints and private credentials are excluded.

Final round: RAPTOR 87/300 (29.0%), flat 81/300 (27.0%), truncated 31/300 (10.3%). RAPTOR minus flat +2.0 percentage points, paired 95% CI [-2.0, 6.0], story-cluster CI [-2.3, 7.0]; exact McNemar p=0.417692 with 22 versus 16 discordant pairs. **STILL INCONCLUSIVE after the two-round limit.** Documents exceed the deliberately configured usable context, not the model's native maximum context.
