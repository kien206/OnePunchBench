# OnePunchBench: Initial Experiments

## Summary

Train the prompt on models up to 4B parameters scored by the archived Open LLM Leaderboard v2, then test on 2025–26 releases that we score ourselves under the same protocol.

- **Ground truth.** The six Open LLM Leaderboard v2 benchmarks (IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-Pro) are the fixed reference protocol. The archive gives third-party scores for 4,576 models submitted from June 2024 to March 2025. Newer models are scored with the same lm-eval `leaderboard` tasks, after we check that we reproduce archived scores.
- **Models.** The search runs on models of 4B or less, which fit a 40GB GPU with room to swap many models per step. The readout and the evaluation extend to 14B (forward pass only). The pool covers dense transformers, hybrids (Mamba-2, Gated DeltaNet, convolution), MoE and pure recurrent models.
- **Splits.** Families are split into train, monitor, calibration and test. Hybrid, recurrent and MoE architectures, several tokenizers, and all 2025–26 releases are kept out of training, so each is its own test.
- **First runs.** Pipeline and ground-truth checks, then fixed-prompt baselines as a go/no-go gate, then gradient-free search, then the bilevel search of the method section, then a short list of ablations.
- **Budget.** About 500 GPU-hours on one 40GB GPU, about three weeks. Scoring the new models ourselves is the largest single item.

## Ground truth

Use the archived Open LLM Leaderboard v2 as training ground truth, and its exact protocol for everything newer. It is the only third-party source we found with thousands of small models, including fine-tunes and merges. That gives the family structure our splits need. Hugging Face retired the leaderboard on [2025-03-13](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/1135), so nothing released later is in it.

| Source | Benchmarks | Models | 2025–26 releases | Role here |
| --- | --- | --- | --- | --- |
| [Open LLM Leaderboard v2 archive](https://huggingface.co/datasets/open-llm-leaderboard/contents) | 6, raw and normalized | 4,576 rows, 2,893 still on the Hub | None | Training ground truth |
| Our runs of the same protocol ([lm-eval `leaderboard` tasks](https://openi.pcl.ac.cn/liux/lm-evaluation-harness/src/branch/main/lm_eval/tasks/leaderboard/README.md)) | Same 6, same shot counts | Any we run | Yes | Ground truth for new models, after a reproduction check |
| [Artificial Analysis](https://artificialanalysis.ai/models/open-source/tiny) | Intelligence Index (10 reasoning and agentic evals) | 49 open models of 4B or less, [142 of 4–40B](https://artificialanalysis.ai/models/open-source/small) | Yes, official releases only | Optional rank check on test models; [attribution required](https://artificialanalysis.ai/api-reference/beta) |
| [Epoch AI Benchmarking Hub](https://epoch.ai/benchmarks/use-this-data) | Benchmark CSV, CC BY 4.0 | Small-model coverage not checked | Unknown | Not used in round 1 |

What the [archive's column statistics](https://datasets-server.huggingface.co/statistics?dataset=open-llm-leaderboard/contents&config=default&split=train) mean for us:

- **Availability.** 1,683 of 4,576 models are no longer on the Hub, so their features cannot be computed.
- **Merges.** 1,724 rows are merges. We keep a merge only if all its parents share one root model, and cap models per family.
- **Architectures.** Llama (1,849), Qwen2 (1,308), Mistral (615) and Gemma 2 (248) dominate. Non-transformers are rare: 3 Falcon-Mamba, 4 RecurrentGemma, 2 Hymba, 1 Jamba, 1 RWKV. New architectures therefore need our own runs.
- **Size.** The median is 8B; 3,598 rows are 13B or smaller.
- **Metadata.** Each row records `Base Model`, `Architecture`, `#Params (B)`, `MoE`, `Merged`, `Precision` and `Chat Template`. `Base Model` gives each model's family.

Protocol for our own runs:

- **Tasks.** The lm-eval `leaderboard` group: BBH 3-shot, GPQA 0-shot, MMLU-Pro 5-shot, MuSR 0-shot, IFEval 0-shot (generative), MATH Lvl 5 4-shot (generative, Minerva answer checking).
- **Settings.** bf16. Chat template on for instruct and reasoning models, off for base models, as the archive's `Chat Template` column records. Thinking disabled where the template allows it.
- **Targets.** Raw scores. The leaderboard's normalized scores are an affine map per benchmark, so standardizing makes them equivalent.
- **Reproduction check.** Rerun 12 archived models (two per major family, base and chat) before scoring anything new. Pass if every benchmark lands within two sampling standard errors of the archive. If not, find the setting that differs first.

Small benchmarks cap how well any predictor can do. At 30% accuracy, a 448-question GPQA set has a sampling error of about 2.2 points, against 0.4 for MMLU-Pro's 12,032 questions. We report this noise ceiling next to every error.

## Model pool

Two tiers. **Tier S** (search) holds models of 4B or less, all loaded inside the search loop. **Tier R** (readout and evaluation) adds models up to 14B, which only read the final prompts once. Older families with archive scores do the training. Every 2025–26 release, and every hybrid, recurrent and MoE model, is kept out of training.

| Family | Architecture | Sizes used (B) | Released | Ground truth | Role |
| --- | --- | --- | --- | --- | --- |
| Qwen2 / Qwen2.5 | Dense transformer | 0.5, 1.5, 3, 7, 14 | 2024 | Archive | Train; surrogate group (Qwen tokenizer) |
| Llama 3.1 / 3.2 | Dense transformer | 1, 3, 8 | 2024 | Archive | Train; surrogate group (Llama 3 tokenizer) |
| Gemma 2 | Dense transformer | 2, 9 | 2024 | Archive | Train |
| SmolLM2 | Dense transformer | 0.14, 0.36, 1.7 | 2024 | Archive | Train |
| Phi-3 / 3.5 mini | Dense transformer | 3.8 | 2024 | Archive | Train |
| Mistral 7B | Dense transformer | 7 | 2023 | Archive | Train (Tier R only) |
| Other archive families (Granite 3, StableLM 2, InternLM2.5, EXAONE 3.5, Falcon3, OLMo 2, …) | Dense transformer | ≤ 14 | 2024 | Archive | Monitor and calibration, picked by the census |
| [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) | Hybrid: Gated DeltaNet + gated attention, 3:1 | 0.8, 2, 4, 9 | Feb–Mar 2026 | Ours | Test: new architecture, new tokenizer (248k vocab) |
| [Gemma 4](https://ai.google.dev/gemma/docs/core) | Dense; per-layer embeddings in E2B and E4B | E2B, E4B, 12 | 2026 | Ours | Test |
| [Granite 4.1 / 4.2](https://research.ibm.com/blog/introducing-granite-4-2) | Dense transformer | 3, 8 | Apr / Aug 2026 | Ours | Test |
| [Nemotron 3 Nano 4B](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b) | Hybrid: 21 Mamba-2 + 4 attention layers | 4 | Mar 2026 | Ours | Test: new architecture |
| [LFM2 / LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | Hybrid: 10 gated-convolution + 6 attention blocks | 0.35–2.6 | 2025 / Jan 2026 | Ours | Test: new architecture, 65k vocab |
| [Ministral 3](https://mistral.ai/news/mistral-3/) | Dense transformer | 3, 8, 14 | Dec 2025 | Ours | Test |
| [Olmo 3](https://allenai.org/blog/olmo3) | Dense transformer | 7 | Nov 2025 | Ours | Test |
| [Granite 4.0-H](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) | Hybrid: Mamba-2 + attention, 9:1; Tiny is MoE | Micro 3; Tiny 7 (1 active) | Oct 2025 | Ours | Test: new architecture |
| gpt-oss-20b | MoE, MXFP4 weights | 21 (3.6 active) | 2025 | Ours | Test, if an MXFP4 kernel runs on your GPU |
| [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Dense, part NoPE layers | 3 | 2025 | Ours | Test |
| [Falcon-H1](https://falconllm.tii.ae/falcon-h1.html) | Hybrid: attention + Mamba | 0.5, 1.5, 3, 7 | 2025 | Ours | Test: new architecture |
| Qwen3 | Dense transformer | 0.6, 1.7, 4, 8, 14 | Apr 2025 | Ours | Test: new release, surrogate tokenizer |
| [RWKV-7](https://www.orcarouter.ai/blog/rwkv-7-goose-transformers-support) | Pure recurrent | 0.4, 1.5, 2.9, 7.2 | 2025 | Ours | Test: new architecture |
| Gemma 3 | Dense, local + global attention | 0.27, 1, 4, 12 | 2025 | Ours | Test |
| Phi-4-mini | Dense transformer | 3.8 | 2025 | Ours | Calibration: unseen tokenizer |
| OLMoE, Qwen1.5-MoE | MoE | 7 (1 active); 14.3 (2.7 active) | 2024 | Archive | Test: held-out architecture |
| Falcon-Mamba, RecurrentGemma | Pure recurrent | 7; 2, 9 | 2024 | Archive | Test: held-out architecture |

- **Fine-tunes.** Each new base model of 4B or less gets up to five popular fine-tunes from its Hub model tree, excluding merges. This gives new families the within-family spread that the archive families already have.
- **Multimodal models.** Qwen3.5, Gemma 4 and Ministral 3 are vision-language models; we load the language model and feed text only.
- **Loading.** RWKV-7 needs `trust_remote_code` and flash-linear-attention. gpt-oss-20b ships in MXFP4; a bf16 copy needs about 42GB, so it runs only if an MXFP4 kernel supports your GPU.
- **Out of round 1.** Gemma 4 26B A4B (57.7GB in bf16), Gemma 4 31B, Meta's Muse Glimmer 30B and Qwen3.6 35B-A3B need 8-bit weights or a bigger GPU. Diffusion LMs give no left-to-right log-probabilities.

Rows without a link are archive families or well-known 2024–25 releases; the census in E0 confirms their sizes and which tokenizers are shared.

## Splits

Split by family: archive families train, monitor and calibrate, and everything released after the archive is test. A family is one pretraining run and everything derived from it, traced through the archive's `Base Model` chain or the Hub's `base_model` tree. Llama 3.2 1B and 3B were distilled from Llama 3.1 8B, so Llama 3.x counts as one family.

| Split | Families | Ground truth | Used for |
| --- | --- | --- | --- |
| Train | Qwen2.5, Qwen2, Llama 3.x, Gemma 2, SmolLM2, Phi-3, Mistral 7B, plus archive families with at least 3 models on the Hub | Archive | Search episodes (4B or less) and the final readout (up to 14B) |
| Monitor | 2–3 archive families with at least 10 models each, e.g. Granite 3 and InternLM2.5 | Archive | Early stopping only |
| Calibration | 2 archive families, plus Phi-4-mini and its fine-tunes (a tokenizer no training family uses) | Archive and ours | Error bars only |
| Test A: held-out archive families | e.g. EXAONE 3.5, Falcon3, OLMo 2, StableLM 2 | Archive | Transfer to unseen families of the same era |
| Test B: new releases | Every 2025–26 family in the pool except Phi-4-mini | Ours | The headline result |
| Test C: held-out architectures | All hybrid, recurrent and MoE models, archive and new | Both | Transfer to unseen architectures |

The test results are also sliced three ways, without new models:

- **Tokenizer.** Surrogate tokenizer (Qwen3 shares Qwen2.5's), seen in training but not a surrogate, or never seen (Qwen3.5, LFM2, RWKV-7, Falcon-H1, Gemma 3 and 4).
- **Size.** Test models of 4B or less, against 7–14B models that are larger than anything in the search.
- **Within family.** Rank correlation among fine-tunes of one base model, where parameter count says nothing.

The final assignment of archive families to monitor, calibration and Test A waits for the census in E0. A stricter variant holds out every family of one organization (for example all Qwen generations), because successive generations share data and tokenizers.

## Compute on one 40GB GPU

One 1,000-step search costs about 18 GPU-hours, and the whole first round about 500. All figures are estimates from FLOP counts and load times; E0 measures the real ones. bf16 weights take 2 bytes per parameter.

| Use | Size limit | Why |
| --- | --- | --- |
| Gradient pass (surrogate models) | 8B; round 1 uses 3B or less | Weights plus activations of a ~500-token prompt; no weight gradients are kept |
| Search loop (exact check) | 4B | About 60 model loads per step |
| Readout and evaluation | 14B (28GB) | One forward pass per prompt |
| 8-bit weights | About 30B | Round 2 only, since quantization changes the features |

Starting configuration for the search:

- **Prompt.** At most 2,048 bytes (about 500 tokens), cut into d = 16 chunks of 128 bytes.
- **Step pool.** 12 training families × 4 models = 48 models per step, drawn from families with at least 4 models of 4B or less. The pool is redrawn every 10 steps, while its 8 episodes are redrawn every step: 8 families inner and 4 outer, so N_in = 32 and N_out = 16. A quarter of episodes are tokenizer-disjoint.
- **Candidates.** C = 64 swaps from a shortlist of 256 tokens per position. All candidates are screened on one model per family; the best 8 get the full check.
- **Length.** 1,000 steps, monitor loss every 50, stop after 6 checks without improvement.

Per step, the exact check is about 1,200 model-prompt passes (~0.6M tokens, ~25 s on 2B models). The 60 model loads take about 20 s from pinned host RAM, and the gradient pass on 6–9 surrogate models about 10 s. That is roughly one minute per step. Holding the current pool needs about 200GB of host RAM; the next pool is prefetched from NVMe in the background.

| Item | Units | GPU-hours each | GPU-hours total |
| --- | --- | --- | --- |
| Ground-truth reproduction (E0) | 12 models | ~1.5 | ~20 |
| Our ground truth, new models of 4B or less | ~60 models | ~1 | ~60 |
| Our ground truth, new models of 7–14B | ~25 models | ~2.5 | ~60 |
| Features for fixed and final prompts | ~400 models × ~16 prompts | Load-bound | ~20 |
| Search runs (T2, T3: 4 configurations × 3 seeds) | 12 runs | ~18 | ~215 |
| Ablations (T4, 1 seed) | 7 runs | ~18 | ~125 |
| **Total** |  |  | **~500** |

Storage: Tier S stays on local NVMe (about 250 models, ~1TB). Tier R is streamed: download a model, score every pending prompt in one pass, delete it. That is about 4TB of downloads per pass, so final prompts are batched into one pass at the end.

## Training experiments

Five steps, each answering one question before the next spends compute. The thresholds in the last column are proposals to agree on before running.

| ID | Question | Setup | Pass criterion |
| --- | --- | --- | --- |
| E0 | Is the pipeline sound? | Ground-truth reproduction on 12 archive models. Feature checks: chunk features sum to the prompt's log-probability; features match across batch sizes, repeat runs, and vLLM vs. transformers. Family census and final split. | Reproduction within 2 sampling SEs; run-to-run feature noise under 1% of the spread between models |
| T1 | Does a fixed prompt say more than model size? | Ridge on fixed prompts, leave-one-family-out over train and monitor families, all Tier R models (list below) | Gate 1: the chunked generic passage beats log parameter count by at least 0.05 mean Spearman, CI over families excluding 0; its error is at least 1.5× the noise ceiling, so search has room |
| T2 | Can search on the exact objective improve a prompt? | Gradient-free search on Tier S, episodes as in the method, started from the generic passage: (a) evolutionary token edits; (b) LLM-proposed rewrites, OPRO-style, from a model outside the test families | Gate 2: monitor loss at least 10% below the starting prompt, and lower error on Test A |
| T3 | Does the surrogate hypergradient help? | Algorithm 1 at T2's compute: (a) one surrogate group, Qwen models of 3B or less; (b) rotating Qwen and Llama 3 groups | Gate 3: T3 reaches T2's best monitor loss in half the GPU-hours, or ends lower at equal GPU-hours |
| T4 | Which parts matter? | Seven ablation runs at one seed (list below) | Reported, no gate |

T1 baselines, each scored with the same ridge readout (prompts are 2,048 bytes):

1. Log parameter count alone (no prompt).
2. A generic passage at d = 1, i.e. bits per byte, the scalar of [Huang et al. (2024)](https://arxiv.org/abs/2404.09937).
3. The same passage at d = 16, close to the log-likelihood vectors of Oyama et al. (2025).
4. Random bytes, and random tokens of one tokenizer.
5. A hand-written probe mixing math, code, instructions, facts and several languages.
6. Item 3 plus log parameter count.

T4 ablations, against the T3 winner:

1. Joint single-level loss instead of the bilevel objective.
2. One fixed train/validation split instead of episodes.
3. No tokenizer-disjoint episodes.
4. d = 1 and d = 64 instead of 16 (two runs).
5. Random token swaps instead of the gradient shortlist.
6. Best of N random prompts, with as many loss evaluations as one search.

Deferred to round 2: holding W fixed in the backward pass, uniform instead of per-family sampling, and adding new releases to training.

## Evaluation experiments

Every prompt (the T1 baselines, best-of-N, and the winners of T2–T4) gets one scoring pass over all test models. A readout refit on all training families then predicts their six scores.

| ID | Test set | What it shows | Ground truth |
| --- | --- | --- | --- |
| V1 | Test A: held-out archive families | Transfer to unseen families of the same era | Archive |
| V2 | Test B: 2025–26 releases, reported per architecture type | Transfer forward in time; the headline result | Ours |
| V3 | Test C: hybrid, recurrent and MoE models | Transfer to architectures the search never saw | Both |
| V4 | Test results by tokenizer: surrogate, seen, never seen | Whether the prompt is tied to the surrogate vocabularies | Both |
| V5 | 7–14B test models, with a readout fit on 4B-or-less models only, and on all sizes | Extrapolation in size | Both |
| V6 | Fine-tunes of one base model (families with at least 4) | Ranking where parameter count is useless | Both |
| V7 | 8-bit and 4-bit copies of ~10 test models | Predictions should move no more than real scores; we score 4 quantized pairs ourselves (about 8 GPU-hours) | Ours |
| V8 | Longest n-gram overlap between each final prompt and the six test sets | Rules out a contamination shortcut | None needed |
| V9 | Optional: 2025–26 official instruct models also on Artificial Analysis | Rank agreement under a different protocol; Spearman only | Artificial Analysis |

Metrics, per benchmark and averaged over the six:

- **Spearman** between predicted and true scores.
- **RMSE** in raw score points, shown next to the noise ceiling from the sampling error.
- **Within-family Spearman**, for V6.

All come with 95% bootstrap intervals that resample families, not models, so large families do not narrow the intervals. Every test set reports the T1 baselines and best-of-N alongside the searched prompts.

## Order of work

Steps run top to bottom. A failed Gate 0, 1 or 2 stops the line until it is fixed; a failed Gate 3 only changes which search the paper reports.

1. **E0 · Pipeline and census.** Reproduce archive scores; check features.
    - Gate 0: do our scores reproduce the archive?
    - If no: stop and fix the protocol. Find the setting that differs (backend, chat template, harness version).
    - If yes: go on to T1, and start scoring the new releases in parallel (step 2).
2. **In parallel, from Gate 0: our ground truth.** Score the 2025–26 releases with the same protocol (about 120 GPU-hours). It must finish before V2–V7.
3. **T1 · Fixed-prompt baselines.** Size, bits per byte, generic passage, hand-written probe.
    - Gate 1: is there signal beyond model size, and room for search to improve?
    - If no: stop and rethink. No search until a fixed prompt beats model size and leaves room to gain.
4. **T2 · Gradient-free search.** Evolutionary token edits and LLM-proposed rewrites.
    - Gate 2: does the search beat its starting prompt?
    - If no: stop and look for overfitting. Compare episode loss with monitor loss.
5. **T3 · Bilevel search.** One surrogate group, then rotating groups.
    - Gate 3: does it beat T2 per GPU-hour?
    - If no: keep T2's search, report it as the method, and run the T4 ablations on it instead.
6. **T4 ablations and V1–V9 evaluation.** One scoring pass for all final prompts.

## Open questions

- [ ] Hardware: is it one A100 40GB? How much host RAM (the step pool needs ~200GB) and local NVMe (~1TB)?
- [ ] Agree on the gate thresholds in the training table before T1 runs.
- [ ] Keep all six benchmarks, or drop GPQA if its spread across small models is close to its sampling error? Decide after E0.
- [ ] Merges: keep single-root merges, as planned, or drop them all?
- [ ] Reasoning models: score with thinking off only (as planned, matching the archive), or also with thinking on as a second target?
- [ ] Scoring the new releases ourselves costs ~120 GPU-hours. Start Test B with official checkpoints only (~40 models) and add fine-tunes later?
- [ ] T2b optimizer: a local instruct model or an API model?
- [ ] One news site (Jan 2026) reports an "Open LLM Leaderboard v3" on Hugging Face. I could not confirm it from a primary source; if it exists, it could give third-party scores for newer models.

## Sources

- Ground truth: [leaderboard retirement post](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/1135) · [archive dataset](https://huggingface.co/datasets/open-llm-leaderboard/contents) · [archive column statistics](https://datasets-server.huggingface.co/statistics?dataset=open-llm-leaderboard/contents&config=default&split=train) · [v2 design blog](https://open-llm-leaderboard-blog.static.hf.space/) · [lm-eval leaderboard tasks](https://openi.pcl.ac.cn/liux/lm-evaluation-harness/src/branch/main/lm_eval/tasks/leaderboard/README.md) · [Artificial Analysis tiny](https://artificialanalysis.ai/models/open-source/tiny) and [small](https://artificialanalysis.ai/models/open-source/small) open models · [Artificial Analysis API terms](https://artificialanalysis.ai/api-reference/beta) · [Epoch AI data](https://epoch.ai/benchmarks/use-this-data)
- Models: [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B) · [Qwen3.5 small models](https://artificialanalysis.ai/articles/qwen3-5-small-models) · [Gemma 4 overview](https://ai.google.dev/gemma/docs/core) · [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Granite 4.2](https://research.ibm.com/blog/introducing-granite-4-2) · [Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) · [Nemotron 3 Nano 4B](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b) · [LFM2.5-1.2B card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) · [LFM2.5 release](https://www.marktechpost.com/2026/01/06/liquid-ai-releases-lfm2-5-a-compact-ai-model-family-for-real-on-device-agents/) · [Mistral 3](https://mistral.ai/news/mistral-3/) · [Olmo 3](https://allenai.org/blog/olmo3) · [SmolLM3 card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) · [Falcon-H1](https://falconllm.tii.ae/falcon-h1.html) · [RWKV-7 loading](https://www.orcarouter.ai/blog/rwkv-7-goose-transformers-support) · [Meta open-weight models](https://mungomash.com/ai/llama/versions/) · [Small models of 2026](https://www.turingpost.com/p/slmslist)
- Baseline: [Huang et al. (2024), Compression Represents Intelligence Linearly](https://arxiv.org/abs/2404.09937)