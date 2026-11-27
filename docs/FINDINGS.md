# Speculative decoding on a consumer CPU: what transfers and what doesn't

Cascade, final findings, 2026-11-26.

Cascade runs local models on one 4-core / 8-thread laptop CPU, no GPU, with llama.cpp through
llama-cpp-python 0.3.36. The target model is qwen2.5-coder 7B at Q4_K_M; the draft and small-answer
model is qwen2.5 0.5B instruct. Every speed result below compares exact greedy output against plain
decoding, token by token. Each finding points to the addendum in
[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) that holds the measurement, and the scripts are in
`scripts/` and `scripts/analysis/`.

## What Cascade achieves

| | Result | Where |
|---|---|---|
| Lossless speed-up, 7B with the 0.5B as draft | 1.49–1.59x overall (grounded 1.86–1.98x, open-ended 1.12–1.27x) | Add. 17, 25 |
| Whole 34-question session (cache, router, small model, drafting) | 2.19x faster, 34/34 correct | Add. 11, 21, 24 |
| 7B resident memory | 7.21 → 4.59 GB, from not memory-mapping repacked weights | Add. 7 |
| Semantic cache with the 7B judging close matches | Rewordings served 22/55 → 54/55, near-misses served 0/59 | Add. 24 |
| One 0.5B as both draft and small-answer model | Small-model tier at no extra RAM | Add. 12 |

## Findings

**1. Draft length should follow this machine's measured verify-cost curve, not a smooth model.**
On a CPU, verifying N draft tokens does not cost a smooth amount. The repacked Q4_K kernels work in
groups of 4 rows, so 4 tokens can cost less than 3, and 16 less than 15. The measured cost relative to
one token is 1.00, 1.38, 1.77, 1.45, 1.94 … (Add. 17, 27).

A cost-aware policy picks the draft length L that maximises (1 + E[accepted]) / C(L), where C is the
measured table. Exact replay compares cost models (`scripts/analysis/cost_shape_replay.py`):

| cost model | code | open |
|---|---:|---:|
| measured table | **2.163x** | **1.532x** |
| monotone (isotonic) fit | 2.095x | 1.520x |
| affine (the smooth model papers assume) | 1.882x | 1.351x |
| flat (GPU-like) | 1.784x | 1.251x |

The measured table beats the affine model by 15% on code and 13% on open-ended text. It is measured
once per model, thread count, CPU and llama.cpp version, takes about 25 s, and is cached.

**2. On this laptop the 7B is limited by memory and unpacking work together, not by bandwidth alone.**
It reads about 15.6 GB/s.
- Q3_K_S reads 25% fewer bytes than Q4_K_M and decodes no faster; IQ4_XS is smaller and slower.
  Only Q2_K is faster (Add. 27).
- Ollama's AVX-512 build decodes the same file only about 5% faster.
- More threads speed up single-token decode in isolation (8 threads −9%), but are slower in the full
  pipeline.

**3. So ideas that assume spare compute fail on a CPU, even when they are new.**
- *Shadow drafting* runs the draft on the spare hyperthreads while the target verifies. It looked free
  in a test that pinned the target's batch threads to 4. At the runner's real settings the target's
  batch verifies already use all 8 logical threads, and the concurrent draft slows each verify by
  14–36%. Live: 1.38x vs 1.49x for the shipped scheme (Add. 25). This contradicts the intuition that
  decoding leaves SMT siblings idle, and it agrees with SubTernary (arXiv 2610.28671), which gives all
  threads to the active model.
- *Layer-skip self-speculation* (LayerSkip, SWIFT, CLaSp) needs a draft that agrees with the target
  72–94% of the time at 30–60% of its cost. The 7B keeping 17 of 28 blocks agrees on 28% of open text
  (Add. 26).
- *A certified sparse output head* computes the exact greedy token without all 152k logits, using
  inner-product bounds. The head is 9–15% of a step, but the winner sits only 0.2·|h| above the mean
  logit, so every bound family leaves 98.5–100% of the rows to compute (Add. 27).

**4. Batching has no cheap regime here.**
Per-token verify cost falls to about 0.25 of a single step at 32–256 tokens and stays there (Add. 27).
Even perfect drafts could give at most about 4x. Prompt processing runs at about 17–20 tokens/s;
Ollama's runs at 9 tokens/s on a fresh prompt.

**5. The drafting rule is close to optimal, and the remaining gap is information.**
Exact replays (Add. 22, 28):
- Optimising draft length for long-run throughput (Dinkelbach) gives −1% to −6%.
- Sibling (tree) drafts give +0.9% on open text.
- Online re-ranking of the draft's top 4 gives at most +1.6 points of top-1 accuracy.
- Blending lookup hints into the draft gives nothing.
- The 0.5B's confidence is already calibrated.

The 7B's token is the 0.5B's first choice 72% of the time on open text but in its top 4 92% of the
time. Closing that gap needs knowledge of the 7B's state, not a better rule.

**6. A cache answer can be reused safely if the big model judges close matches** (Add. 24).
Embedding similarity alone cannot tell rewordings from near-misses ("capital of Austria" after "capital
of Australia"). One yes/no prompt evaluation of the 7B, with no generation, can.

## What we checked for novelty

Ideas that turned out to be published already:
- reusing near-miss answers as drafts: Oilbird, Cacheback;
- learning the draft from rejections: OSD, OnlineSPEC, DVI, OmniDraft;
- hardware-aware trees: Sequoia;
- speculative cascades;
- Big-Little Decoder;
- BitStack and any-precision weights;
- thermal-aware on-device inference: EnerInfer, SERENO;
- semantic KV reuse: CacheBlend;
- pre-answering predicted questions: InstCache, StorInfer;
- difficulty routing: RouteLLM.

The ideas that were new to our search (shadow drafting on SMT threads, exact head pruning) are the ones
these measurements refute on this hardware. What is new here is empirical: the measured, non-monotone
cost curve and how much it matters, and why compute-sharing ideas fail on a laptop CPU.

## EAGLE-style drafting from the 7B's own state

The one direction with headroom left is a draft that reads the 7B's own state, as in EAGLE. It was
tested in two gates (`scripts/analysis/eagle_gate.py`):

1. **Access.** llama-cpp-python returns the 7B's final hidden state h per position together with the
   logits (`embedding=True`, pooling none). W·h reproduces the logits, with the same winning token at
   every position. Feasible.
2. **Information.** The 7B ran over 30,336 tokens of local Python source and Markdown, keeping h, its
   greedy token and the 0.5B's top 4. On held-out documents, the closed-form EAGLE-like map
   h_{t+1} ≈ A·h_t + B·e(x_{t+1}) predicts the 7B's next token less well than the 0.5B, and re-ranking
   the 0.5B's top 4 with it adds at most half a point:

| | 0.5B top-1 | linear predictor | 0.5B top 4 re-ranked |
|---|---:|---:|---:|
| code | 71.9% | 28–30% | 72.1–72.4% |
| prose | 69.6% | 24–27% | 69.7–70.0% |

A real EAGLE head needs a trained decoder layer (about 0.23B parameters at 7B width, plus the output
head) and millions of tokens of the 7B's features. At about 20 tokens/s of 7B prompt processing, that is
about 6 days of compute on this laptop before any training. Not pursued.

## Conclusion

On a 4-core laptop CPU, Cascade gets the gains that exist:
- 1.5–1.6x lossless decoding (about 1.9x on grounded and code prompts);
- a 2.19x faster mixed session with every answer correct;
- the 7B's resident RAM nearly halved.

The rest of the search space was checked. It is either published (and already used or not worth it
here) or measured to fail on this hardware. Further speed would need different hardware (a GPU, more
cores, more memory bandwidth) or GPU-scale training of a target-specific draft head. The project is
closed at this point.
