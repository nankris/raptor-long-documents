# Does a RAPTOR summary tree beat plain chunks? Not for QuALITY with a 4B reader

I tested RAPTOR-style retrieval against flat 100-word chunks on 145 QuALITY questions over 8 long articles. Both arms got the same ~1,500-token context and the same Qwen3-4B reader.

Result: a dead tie at 66.9%. Zero context scored 39.3% and the whole article scored 75.2%, so retrieval recovers most of the gap but not all of it.

Building the trees took 245 s (82 summarisation calls) on an L4. That's real overhead for no measured gain. The summaries did enter the top-k, but they displaced chunks rather than adding information, and QuALITY questions often hinge on details that summaries drop.

Caveats: 145 questions give roughly ±8 points of noise, the articles are only a few thousand words (the full text fits in context), and a 4B model writes mediocre summaries. RAPTOR may pay off for much longer documents or thematic questions. This test can't show that.

Code and raw predictions are in the repo.
