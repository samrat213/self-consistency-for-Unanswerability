# CRP-family experiments on VQA unanswerability

Qwen3.8-27B, **answerability-aware** prompts (the abstain option is offered), 20 temperature samples per question.
Datasets: UVQA (dev 52 / test 448), UNK-VQA (200 / 900), SpatialUncertain (200 / 900), all 50% answerable.
Everything is tuned on dev (Overall F1 = mean of answerable-F1 and unanswerable-F1) and reported once on test.
Run 2026-10-30. All tables: [`RESULTS_raw.md`](RESULTS_raw.md); numbers: `results_vqa.json`.

| file | what |
|---|---|
| `data/<ds>_aware.json` | the 20-sample votes in the `experiments_CRP.py` item format + fixed dev/test split |
| `experiments_vqa.py` | E1–E5 below (imports `../experiments_CRP.py`; numpy + scipy) |
| `make_report.py` | `results_vqa.json` → `RESULTS_raw.md` + `plots/sequential.png` |
| `export_crp_data.py` | how `data/` was made (needs the VQA codebase in `~/vqa-datasets`, reference only) |

Reproduce: `python vqa/experiments_vqa.py && python vqa/make_report.py` (~10 min CPU).

## Background: what the models assume

- **CRP:** each new sample repeats an earlier answer with probability ∝ its count, or gives a new answer with
  probability ∝ α. The likelihood depends on the samples only through the number of distinct answers K, so a
  two-class CRP likelihood ratio is a threshold on K (the same as `CRP_k`).
- **Pitman-Yor (discount d):** each existing cluster gives up d to the "new answer" option. Cluster sizes then enter
  the likelihood through roughly (d_A − d_U)·Σ_k log n_k, i.e. how *evenly* the samples split. "One consensus plus
  noise" (18,1,1) can then differ from "a real split" (7,7,6).
- **Reserved / Hurdle / ZI_PYP:** "I don't know" (IDK) answers go to a reserved table whose rate gets its own
  Beta-Binomial per class.
- **Colab notebook:** α itself varies by question, α ~ Normal(μ_Z, σ_Z) per class. It is simulated only; the
  sweeps assume answerable and unanswerable questions differ in α.

## Results

### E1: `experiments_CRP.py` as-is on VQA (test Overall F1)

| method | UVQA | UNK-VQA | SpatialUncertain |
|---|---|---|---|
| self-consistency | 76 | 42 | 51 |
| self Consistency + T / scUn | 76 / 75 | 40 / 40 | 59 / 59 |
| CRP_k | 76 | 41 | 57 |
| Adaptive Consistency | 73 | 40 | 54 |
| TwoRest_CRP | 47 (answers 29%) | 41 | 57 |
| TwoRest_Reserved_CRP | 79 | 41 | 57 |
| **TwoRest_Hurdle_CRP** | **80** | 40 | 57 |
| **TwoRest_ZI_PYP** | **80** | 36 (answers 10%) | **59** |
| *Oracle_Intermediate (upper bound)* | *87–88* | *50–54* | *72–74* |
| best rule from our earlier VQA dashboard rows | 78.2 (AC 95%) | 42.7 (CRP_k) | 59.1 (Unans votes + CRP_k) |

Whole numbers because `experiments_CRP.calculate_metrics` rounds to 2 decimals.

- **UVQA:** the IDK-aware two-restaurant models are the best method so far (+2 over our best rule, +4 over SC).
- **UNK-VQA:** no gain. ZI_PYP's dev-tuned threshold nearly stops answering.
- **SpatialUncertain:** ZI_PYP ties our best rule, and Hurdle is 2 points behind.
- **TwoRest_CRP (cluster count only) is never better than CRP_k** and collapses on UVQA, as the theory predicts.
- **The fitted discount is ≈ 0** on UVQA and SpatialUncertain (d ≤ 0.005), so ZI_PYP ≈ Hurdle there. On UNK-VQA,
  d_A ≈ d_U (0.195 vs 0.185). **The gains come from the IDK table, not from the CRP/PYP part.**
- The oracle leaves a lot of room (+7 / +10 / +14), but that room is in questions it decides using the label.

### E2: how many samples are needed? (sequential stopping, test, mean over 10 sample orders)

SPRT = after each sample, abstain if the Hurdle LLR ≥ A, answer if ≤ B, else draw another; (A, B) chosen on dev
for each average-sample budget. Plot: [`plots/sequential.png`](plots/sequential.png).

| | UVQA F1 (avg n) | SpatialUncertain F1 (avg n) |
|---|---|---|
| 1 sample | 73.2 | 50.5 |
| SPRT Hurdle, budget ≤ 4 | 78.2 (3.4) | 57.9 (3.7) |
| SPRT IDK-only, budget ≤ 5 | 78.9 (4.5) | 57.3 (4.5) |
| Adaptive Consistency, budget ≤ 4 | 73.2 (1.0) | 52.4 (3.6) |
| fixed n = 3, Hurdle LLR | 78.4 | 57.3 |
| fixed n = 20, Hurdle LLR | 80.0 | 59.3 |

- **3–5 samples recover all but about 1–1.5 F1 of the 20-sample result** (4–6× cheaper).
- Adaptive Consistency is much worse at small budgets: it only stops when answers agree, and never uses the IDK votes.
- **The SPRT adds little over a fixed small n.** Most of the benefit is that ~3 samples already carry the signal.
- UNK-VQA is flat at 40–42 for every method and budget, so there is nothing to save there. Its references are noisy
  and the model over-abstains (see E4).

### E3: is there anything for the Pitman-Yor discount to use? (dev)

Within a fixed K, the AUROC of evenness Σ log n_k for unanswerable is 0.25–0.59: no consistent direction, and below
0.5 when IDK votes are dropped. At the same K, the number of IDK votes gives 0.53–1.00. **No.**

### E4: hard vs unanswerable (dev)

easy = answerable and the self-consistency answer is right; hard = answerable and it is wrong or abstains.

| AUROC | UNK-VQA unans vs hard | SpatialUnc unans vs hard | SpatialUnc unans vs easy |
|---|---|---|---|
| #IDK votes | **0.41** | 0.58 | 0.73 |
| K / top share / evenness | 0.53 / 0.53 / 0.53 | 0.42 / 0.44 / 0.46 | 0.65 / 0.65 / 0.63 |
| Hurdle LLR | 0.54 | 0.54 | 0.73 |

(UVQA has only 4 hard dev questions.) **Every sampling signal separates unanswerable from easy, but none separates
unanswerable from hard (all 0.41–0.58).** The samples tell you the model is struggling, not why. On UNK-VQA,
52 of 74 hard answerable questions are ones the model itself abstains on, so the IDK votes even point the wrong way.

### E5: the Colab's hierarchical CRP fitted to real data (dev)

Per-question α estimates: A 0.17 / 0.34 / 0.22 vs U 0.15 / 0.36 / 0.30 (UVQA / UNK / Spatial). The gap is ≈ 0,
an order of magnitude smaller than the simulated μ = 2 vs 6. AUROC is 0.45–0.52 (K alone: 0.48–0.58). Real VQA data
sits in the chance corner of the notebook's heatmap. Two reasons:
1. unanswerable questions mostly get *confident* answers or IDK, not scattered ones (many hallucinations are 20/20);
2. with yes/no or 4-option answers, K ≤ 2–5 whatever the model's uncertainty, so a CRP has no room to spread.

Two notebook issues:
- Cell 2's `classify_question` subtracts each class's own max before exponentiating, which breaks the class ratio
  (cell 3 is correct).
- `train()` averages α under a different prior per class (N(5,2) vs N(2,1)), which creates part of the fitted gap
  (UVQA raw gap −0.03 → fitted 0.90 vs 0.81).

## Takeaways

1. On VQA, the useful part of the two-restaurant models is the **IDK-rate model**. The CRP/PYP clustering part adds
   ~nothing, because cluster count and evenness carry almost no answerability signal here.
2. **A few samples suffice:** with an IDK-aware rule, 3–5 samples reach within ~1–1.5 F1 of 20.
3. **Hard vs unanswerable is not separable from samples alone.** That needs a different kind of signal.

## Suggested next steps

1. **Run the same E2 sequential analysis on the math datasets** (free-form answers, so a CRP is more natural there)
   to see whether the CRP part helps when the answer space is open. Needs the math sample files.
2. **Replace the CRP with a Dirichlet-multinomial over the fixed options** for MCQ / yes-no data.
3. **Add signals from outside sample agreement** for confident hallucinations and for hard vs unanswerable:
   - token log-probs (exist for VQA dev);
   - P(True);
   - question paraphrase / image perturbation;
   - a hidden-state linear probe (the model is run locally, so activations are available).
4. **Report "F1 vs average samples" curves** (plus AURC) as a main result, with Adaptive Consistency /
   Early-Stopping SC as baselines.
