## Validation results

### Spec 03: Qwen3-8B on rtx3090-oct-22 deployment configuration (kiss spec)

**Commit**: 1e932ee068c781ff71d41c05d87da536e75b28f8

Full validation, profile `rtx3090_oct_22`, scope `full`, collection `merge_quality__merge_quality_easy` (100 cases).

#### AC1

**Status**: accepted

| Validation metric | Computation status | Acceptance status | Computed value | Expected value or threshold | Error message |
|-------------------|--------------------|-------------------|----------------|-----------------------------|---------------|
| VM1 | computed | accepted | 0.1317 s | 0.5 s | — |

#### AC2

**Status**: accepted

| Validation metric | Computation status | Acceptance status | Computed value | Expected value or threshold | Error message |
|-------------------|--------------------|-------------------|----------------|-----------------------------|---------------|
| VM2 | computed | accepted | 0.1521 s | 1.0 s | — |

#### AC3

**Status**: accepted

| Validation metric | Computation status | Acceptance status | Computed value | Expected value or threshold | Error message |
|-------------------|--------------------|-------------------|----------------|-----------------------------|---------------|
| VM3 | computed | accepted | 0.007764 s | 0.01 s | — |

#### AC4

**Status**: accepted

| Validation metric | Computation status | Acceptance status | Computed value | Expected value or threshold | Error message |
|-------------------|--------------------|-------------------|----------------|-----------------------------|---------------|
| VM4 | computed | accepted | 0.007797 s | 0.05 s | — |

#### AC5

**Status**: accepted

| Validation metric | Computation status | Acceptance status | Computed value | Expected value or threshold | Error message |
|-------------------|--------------------|-------------------|----------------|-----------------------------|---------------|
| VM5 | computed | accepted | 14.484 GiB | 16 GiB | — |

#### AC6

**Status**: valid (rejected)

| Validation metric | Computation status | Acceptance status | Computed value | Expected value or threshold | Error message |
|-------------------|--------------------|-------------------|----------------|-----------------------------|---------------|
| VM6 | computed | rejected | 5.539 % | 1 % | — |

The 4-bit release costs accuracy against the bf16 baseline of the profile. The result is reproducible: a second full validation of the same commit returned the same VM6 to 15 significant digits (5.538932890463419 %). The loss is diffuse extraction quality: against the baseline, recall drops from 0.696 to 0.642 and completeness from 0.756 to 0.725, while precision drops from 0.333 to 0.305.

The bf16 release cannot replace the 4-bit one: it rejects AC3 (VM3 0.020227 s against the 0.01 s threshold) and AC5 (VM5 21.99 GiB against the 16 GiB threshold), because its weights alone (15.27 GiB) exceed the peak-VRAM budget before any KV cache is allocated, and moving them across this card's 936 GB/s costs at least 17.5 ms per output token.


### Spec 04: Qwen3-8B accuracy recovery

Full validation on `rtx3090-oct-22`, profile `rtx3090_oct_22`, collection `merge_quality__merge_quality_easy` (100 cases). The validator, baseline, data, prompts, scorer, and thresholds are unchanged. Negative VM6 means the measured score exceeds the original-precision baseline.

Thresholds: VM1 < 0.5 s; VM2 < 1 s; VM3 < 0.01 s; VM4 < 0.05 s; VM5 < 16 GiB; VM6 < 1%.

| Candidate | Commit | VM1 s | VM2 s | VM3 s | VM4 s | VM5 GiB | VM6 % | Criterion statuses AC1-AC6 |
|---|---|---|---|---|---|---|---|---|
| AWQ baseline | `beff9e6d7a1237d8a97e2d0769f2587ede243539` | 0.131888927499 | 0.1521738266 | 0.0077648363458 | 0.00779361894631 | 14.484375 | 5.53893289046 | accepted, accepted, accepted, accepted, accepted, valid |
| autoround_gptq | `005429d7f711ab470882defd3dad14cbb9049e47` | 0.133820807 | 0.15271554385 | 0.00771498426428 | 0.00773558405514 | 13.48046875 | 3.52748513074 | accepted, accepted, accepted, accepted, accepted, valid |
| autoround_greedy | `444771188ad063798f0954d4b069c5e097e3b0ba` | 0.1317705295 | 0.151847791851 | 0.00756212122215 | 0.00758691249198 | 13.48046875 | 3.41568842444 | accepted, accepted, accepted, accepted, accepted, valid |
| gptq8_draft2 | `5f8d857c88d93fb5e358dc919bd7ee615a9280e7` | not computed | not computed | not computed | not computed | not computed | not computed | invalid, invalid, invalid, invalid, invalid, invalid |
| gptq8_draft2_compile_fixed | `88c604f94920fa37c45a8b07536b71003864e1eb` | not computed | not computed | not computed | not computed | not computed | not computed | invalid, invalid, invalid, invalid, invalid, invalid |
| gptq8_draft2_context8k | `1f1cae42efe2641ae56b14ccec033bae315a4110` | 0.1632114605 | 0.1988065675 | 0.00987536141826 | 0.0109831833593 | 13.21875 | 0.636433348929 | accepted, accepted, accepted, accepted, accepted, accepted |
| gptq8_fp16_exllama | `44e45149d4d709e0f42fdb56372ab5dc8f7d2f0f` | not computed | not computed | not computed | not computed | not computed | not computed | invalid, invalid, invalid, invalid, invalid, invalid |
| gptq8_fp16_exllama_context12k | `f64f47900a2b4a9cc9138ea0691c66cdc42425fa` | 0.129560895502 | 0.135961884749 | 0.0111200560225 | 0.011298220094 | 13.28515625 | -0.369665406728 | accepted, accepted, valid, accepted, accepted, accepted |
| gptq8_fp16_marlin_context12k | `16dbce2e0fc83f5ced07711048a36bcf4c23f5f2` | 0.157586104 | 0.20468093645 | 0.0115568851742 | 0.0117367662673 | 13.318359375 | 0.566244712953 | accepted, accepted, valid, accepted, accepted, accepted |
| gptq8_graphs_greedy | `6a900077430be905396ce5c874c018e59b939e0d` | 0.144666290498 | 0.162857120951 | 0.0114742774338 | 0.0115082322061 | 13.318359375 | -0.568607792289 | accepted, accepted, valid, accepted, accepted, accepted |
| gptq_group32 | `ed2374135aabf3221d5e8aae7748284e0599e79d` | 0.1347429545 | 0.153888357099 | 0.00816946497644 | 0.00819867325582 | 13.462890625 | 5.1417975115 | accepted, accepted, accepted, accepted, accepted, valid |
| hybrid_all_greedy | `93be1c1324ba53b00d203efb1d60e1aed67bfdff` | 0.141046401 | 0.157397589851 | 0.0101457251482 | 0.0101757719944 | 13.3125 | 2.73784330477 | accepted, accepted, valid, accepted, accepted, valid |
| hybrid_qkv_greedy | `35e1ea6c8c4a14fead1cf2b7cbb49d56032f6e21` | 0.13933111 | 0.156431204949 | 0.00917970103577 | 0.00921419639411 | 13.30859375 | 2.9910525441 | accepted, accepted, accepted, accepted, accepted, valid |
| junhowie_gptq | `f1b6e08f2b9bb4c8b838abd791528e6562212a35` | 0.1314588845 | 0.1519529414 | 0.00780644529128 | 0.00784203755741 | 13.48046875 | 11.2889046659 | accepted, accepted, accepted, accepted, accepted, valid |
| mixed_gptq8_attention4_mlp | `278c4d7841f6ead40b51f1d55c47145dfab789ff` | 0.135482330001 | 0.155750825302 | 0.00878145468208 | 0.00881228258878 | 13.318359375 | 5.05126297487 | accepted, accepted, accepted, accepted, accepted, valid |
| redhat_w4a16 | `0a1ed1a639ee14af64b3fdee47828dfdc18da4b5` | 0.143440119499 | 0.173514966749 | 0.00785141784927 | 0.00793787063068 | 13.46484375 | 8.00587870391 | accepted, accepted, accepted, accepted, accepted, valid |

#### Selected deployment

`gptq8_draft2_context8k` accepted all six criteria. Serve the pinned JunHowie GPTQ8 target with BF16 activations, Marlin, and the pinned original Qwen3-0.6B draft. The target uses standard rejection to verify proposals; synthetic acceptance is not used. Set two speculative tokens and an 8192-token context. Greedy decoding is an explicit default that callers can override. Runtime startup confirms both models loaded (10.02 GiB model-loading memory), standard rejection, and active draft acceptance metrics. The final integrated commit must be full-validated separately before delivery.

Target: `JunHowie/Qwen3-8B-GPTQ-Int8` at `e131f54dea2ba1f99bbee218f75548ed00646cb9`. Draft: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`. The donor architecture and tokenizer match the original target; a model-card ancestry claim is not independent source-revision provenance. The draft is used only for proposals, not as a substitute for the target.

#### Diagnostic failures

The first Exllama attempt needed more KV cache than available at 24576 tokens. Its corrected 12288-token run passed accuracy but missed median TPOT. FP16 Marlin also passed accuracy but missed median TPOT. Four-bit releases, mixed GPTQ, and restored BF16 attention did not reach <1% loss. Q/K/V restoration was the best fallback preserving AC1-AC5, but is superseded by the accepted speculative candidate. The unused hybrid helper and tests are removed from the final tree; all experiments remain in git history.

The first draft attempt failed because compile size 1 is padded to 3 under two-token speculation. The second fixed compile sizes but needed 3.0 GiB combined KV cache against 2.51 GiB available. Reducing context to 8192 resolved startup. These failures are invalid runs, not measured accuracy or hardware-absence evidence.

The four-bit checkpoint comparisons retain generation defaults. Candidates named greedy, the Exllama/FP16 GPTQ8 runs, and speculative candidates set default temperature to zero. These are joint precision/decoding experiments, not weights-only comparisons. Full validation, not local tests, establishes acceptance. Median TPOT on the accepted candidate is close to its threshold; this result is not a guarantee for other workloads or hardware.

#### Evidence

Exact requests, official responses, setup/deployment logs, and predictions are retained at `/home/ubuntu/accuracy_recovery_20261009/<candidate>/` on the GPU host. Local copies are under `/home/tony/.hermes/cache/scratch/llm_accuracy_evidence/accuracy_recovery_20261009/`. Baseline values are also in `/home/tony/.hermes/cache/scratch/llm_accuracy_baseline.json`.
