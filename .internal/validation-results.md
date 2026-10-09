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
