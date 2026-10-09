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


### Spec 04: Qwen3-8B accuracy recovery (validation in progress)

Full validation on `rtx3090-oct-22`, profile `rtx3090_oct_22`, collection `merge_quality__merge_quality_easy` (100 cases). The validator, baseline, datasets, prompts, scorer, and thresholds are unchanged. A valid criterion is measurable but not accepted.

Thresholds: VM1 < 0.5 s; VM2 < 1 s; VM3 < 0.01 s; VM4 < 0.05 s; VM5 < 16 GiB; VM6 < 1%. Negative VM6 means the measured score exceeds the original-precision baseline.

| Candidate | Commit | VM1 s | VM2 s | VM3 s | VM4 s | VM5 GiB | VM6 % | Criterion statuses AC1-AC6 |
|---|---|---|---|---|---|---|---|---|
| AWQ baseline | `beff9e6d7a1237d8a97e2d0769f2587ede243539` | 0.131888927499 | 0.1521738266 | 0.0077648363458 | 0.00779361894631 | 14.484375 | 5.53893289046 | accepted, accepted, accepted, accepted, accepted, valid |
| autoround_gptq | `005429d7f711ab470882defd3dad14cbb9049e47` | 0.133820807 | 0.15271554385 | 0.00771498426428 | 0.00773558405514 | 13.48046875 | 3.52748513074 | accepted, accepted, accepted, accepted, accepted, valid |
| gptq8_graphs_greedy | `6a900077430be905396ce5c874c018e59b939e0d` | 0.144666290498 | 0.162857120951 | 0.0114742774338 | 0.0115082322061 | 13.318359375 | -0.568607792289 | accepted, accepted, valid, accepted, accepted, accepted |
| gptq_group32 | `ed2374135aabf3221d5e8aae7748284e0599e79d` | 0.1347429545 | 0.153888357099 | 0.00816946497644 | 0.00819867325582 | 13.462890625 | 5.1417975115 | accepted, accepted, accepted, accepted, accepted, valid |
| hybrid_qkv_greedy | `35e1ea6c8c4a14fead1cf2b7cbb49d56032f6e21` | 0.13933111 | 0.156431204949 | 0.00917970103577 | 0.00921419639411 | 13.30859375 | 2.9910525441 | accepted, accepted, accepted, accepted, accepted, valid |
| junhowie_gptq | `f1b6e08f2b9bb4c8b838abd791528e6562212a35` | 0.1314588845 | 0.1519529414 | 0.00780644529128 | 0.00784203755741 | 13.48046875 | 11.2889046659 | accepted, accepted, accepted, accepted, accepted, valid |
| redhat_w4a16 | `0a1ed1a639ee14af64b3fdee47828dfdc18da4b5` | 0.143440119499 | 0.173514966749 | 0.00785141784927 | 0.00793787063068 | 13.46484375 | 8.00587870391 | accepted, accepted, accepted, accepted, accepted, valid |

The four-bit checkpoint comparisons retain their model generation defaults. `gptq8_graphs_greedy`, `hybrid_qkv_greedy`, and the Exllama candidate explicitly set default temperature to zero. These are joint precision/decoding experiments, not weights-only comparisons. Callers can override the default.

The Q/K/V hybrid restores original BF16 attention projections and input-layernorm, while retaining AWQ O, MLP, and post-attention norm. Startup confirms `auto_awq`, Marlin, and a 6.96 GiB weight footprint. It preserves AC1-AC5, but VM6 remains rejected. GPTQ8 with BF16 activations passes AC6, but rejects AC3; this disproves the earlier claim that all quantized weights must fail AC6. No accepted implementation is established by the table.

The first Exllama run is invalid: its 24576-token context needs 3.38 GiB of KV cache, but only 3.33 GiB is available. The response’s generic GPU error does not mean the GPU is absent; the startup log gives the actual failure. A corrected 12288-token candidate is prepared.

Per-candidate requests, official responses, setup/deployment logs, and predictions are retained at `/home/ubuntu/accuracy_recovery_20261009/<candidate>/` on the GPU host. Local evidence is under `/home/tony/.hermes/cache/scratch/llm_accuracy_evidence/`. Baseline values are also retained in `/home/tony/.hermes/cache/scratch/llm_accuracy_baseline.json`.
