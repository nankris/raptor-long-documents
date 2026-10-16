## Sampling and compute audit

The folder already contained a preregistration commit and prepared scripts/data when this continuation began, but no completed statistical round or active runtime. The old usage snapshot showed one allocation without start/stop records, so any earlier CU cannot be reconstructed. This continuation’s named session is `tig074r-longdocs`; GPU NVIDIA L4. Account balances are omitted and other users’ sessions are untouched.

Round 1 used the first six full public NarrativeQA validation sources, 25 questions each. Round 2 retained them and added Cynthia’s Revels, Brewster’s Millions, Beyond the Rocks, Siddhartha, Almayer’s Folly, and The Mysterious Affair at Styles. The metadata upper word bound widened from 60K to 80K before round-2 inference, excluding Goblin Market (poetry collection), Der Judenstaat (non-narrative work), and Cratylus (dialogue). Actual downloaded lengths are independently checked above 20K. This is a purposive eligibility restriction, not a representative random sample of NarrativeQA. The unchanged original six and cached trees preserve the paired comparison across arms.

Minimal-answer instructions in round 2 address observed verbosity penalties under lexical F1. Round 1 and round 2 therefore estimate different prompt/sample configurations; their accuracy changes cannot be attributed solely to additional questions. The rule and scoring threshold remain unchanged, and the two-look α=0.025 remains conservative.

Compute accounting: pending final stop and allocation-duration estimate.


## Final compute audit and stopping outcome
Recovered CLI lifecycle timestamps supersede the earlier statement that prior allocation could not be reconstructed. The first L4 allocation ran from 2026-10-14 23:51:37 UTC to 2026-10-15 00:52:55 UTC (3,678.28 seconds); the second from 00:55:03 to 03:08:56 UTC (8,033.09 seconds). Both termination records say `pruned`, not a confirmed explicit stop. At the previously recorded 1.54 CU/hour, total allocation cost is **approximately 5.01 CU** (1.57 + 3.44), slightly exceeding the 5-CU limit by about 0.01 CU. Actual session billing is unavailable; account-wide balance changes cannot isolate this experiment. This budget/cleanup failure is disclosed, not treated as successful compliance. Model execution itself totaled 5,419.9 seconds across two rounds; allocation includes setup and idle time.

On final audit, `source /workspace/tools/setup.sh`, `colab usage`, and `colab sessions` confirmed zero active assignments and no active sessions. `colab stop -s tig074r-longdocs` returned session not found. No new GPU allocation or third statistical round was started. Sanitized lifecycle evidence is in `results/allocation_audit.json`; account balances, runtime endpoints and private credentials are excluded.

Final round: RAPTOR 87/300 (29.0%), flat 81/300 (27.0%), truncated 31/300 (10.3%). RAPTOR minus flat +2.0 percentage points, paired 95% CI [-2.0, 6.0], story-cluster CI [-2.3, 7.0]; exact McNemar p=0.417692 with 22 versus 16 discordant pairs. **STILL INCONCLUSIVE after the two-round limit.** Documents exceed the deliberately configured usable context, not the model's native maximum context.
