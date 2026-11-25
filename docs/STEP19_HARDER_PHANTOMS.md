# Step 19: Heron run on higher-curvature phantoms

This optional run tests whether Heron p = 1 QAOA keeps its problem-dependent signal on phantoms
where spatial coupling is actually needed. On those phantoms the exact Ising solution beats the
non-spatial orientation baseline. On the four standard phantoms it does not (see `analysis/`).

## Frozen design

| Item | Value |
| --- | --- |
| Backend | `ibm_aachen` (Heron r3), fractional RZZ |
| Angles | frozen hardware-calibrated Heron angles (gamma 0.710398, beta 1.226281), no retuning |
| Primary conditions | 90° centre with c = 50° and 60° centre with c = 60°, seeds 2001–2015 |
| Drift control | standard `curved_90`, seeds 2001–2005 (compare with published Heron values) |
| Circuits | 35 in one SamplerV2 job, 10,000 shots each, XY4 DD, no twirling |
| Transpilation | 20 trials per circuit, lowest calibrated error proxy |
| QPU cap | `max_execution_time` 900 s (job stops rather than overspend) |

Classical reference values (sign-invariant agreement, mean over 15 seeds):

| Condition | MILP | Orientation two-means | Ideal p = 1 gap |
| --- | --- | --- | --- |
| 90°, c = 50° | 0.896 | 0.653 | 0.249 |
| 60°, c = 60° | 0.781 | 0.648 | 0.261 |

Pre-specified endpoints, per primary condition:

- **E1:** Heron mean normalized gap < 1, where 1 is uniform random sampling.
- **E2:** Heron execution penalty, defined as Heron mean gap minus the ideal gap.
- **E3:** agreement of the lowest-energy Heron sample vs the orientation two-means baseline, paired by seed.

Secondary endpoints:

- CVaR-10% gap.
- Lowest-energy Heron sample vs the best of 10,000 uniform random samples.
- The drift control.

All tests are exact two-sided paired sign-flip tests, with Holm correction within each condition.

## Commands

```bash
export PYTHONPATH=src/phantom:src/qaoa:src/hardware/heron
python src/hardware/heron/step19_harder_phantoms.py prepare --out runs/step19            # no QPU time
python src/hardware/heron/step19_harder_phantoms.py submit  --out runs/step19            # validation only
python src/hardware/heron/step19_harder_phantoms.py submit  --out runs/step19 --submit   # spends QPU time
python src/hardware/heron/step19_harder_phantoms.py collect --out runs/step19            # when the job is done
```

Offline check without an IBM account: `prepare --fake` (fake_aachen snapshot), then `simulate`.

Expected cost: 350,000 shots in total. That is roughly 2–3 minutes of QPU time on Heron, about
the size of the earlier replication job (24 circuits × 5,000 shots), so check that job's usage
in the IBM dashboard for the RU equivalent.

## Outcome (job `dbl464hul9ic73fc49ug`, 3 November 2026)

The run used 95 s of QPU time (1.57 RU). E1 and E2 were met and E3 was not. Results, raw counts and
the reproduction notes are in [`data/phantom/step19_hardware`](../data/phantom/step19_hardware/README.md).

The classical reference values above were computed on Linux. On the graphs as executed (macOS), MILP
agreement was 0.888 and 0.773, and two-means agreement was 0.653 and 0.627; see the README for why
regenerated graphs can differ slightly between platforms. Note that this run used 20 transpilation
trials per circuit, whereas the main Heron experiments used 10.
