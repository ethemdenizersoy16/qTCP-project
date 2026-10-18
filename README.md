# qTCP v1.0: Baseline Implementation

This branch holds **qTCP v1.0**, the baseline implementation of the qTCP quantum
transport protocol
[[Yu, Lai & Zhou, IEEE TQE 2021]](https://arxiv.org/abs/1903.10685) as
specified. It runs on the [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe)
quantum network simulator. It accompanies *"qTCP: Implementation, Benchmarking,
and Dynamic Decoding"* by Ethem Deniz Ersoy and Ertan Onur (WINS Lab, Middle
East Technical University), submitted to IEEE ICC 2027.

**The dynamic-decoding variant (v2.0), the full documentation and most of the
raw data are on the [`main`](https://github.com/ethemdenizersoy16/qTCP-project/tree/main) branch.**

## What v1.0 does

Each logical qubit is encoded into five shares with ((3,5)) quantum secret
sharing (QSS). Each share is protected by the [[5,1,3]] quantum error-correcting
code (QEC), forming the leaves. If held plus delivered shares of a non-leaf
block drop to k = 3, Alice recursively QSS-encodes its next share. Leaves are
never recursed on.

v1.0 follows a **fail-stop** policy:

- A leaf goes LOST on its first failed share. There are no restarts.
- QSS layers stop sending as soon as k = 3 shares are delivered.
- Bob always erasure-decodes from three shares.

## Headline result

At tree depth one, v1.0 is worse than bare teleportation at almost every
non-trivial operating point tested. Each attempt costs about 15× the
entanglement (14.5 pairs per attempt).

- **Pure loss:** break-even against bare is at ℓ ≈ 0.032. At ℓ = 0.30, v1.0
  delivers 4% of qubits versus bare's 70%.
- **Pure corruption (ideal gates):** break-even against bare is at ε ≈ 0.034.

v2.0 on `main` shifts these thresholds to ℓ > 0.60 and ε ≈ 0.14–0.18.

## Reproducing the results

The runs use Python 3.12 on Linux.

```bash
git clone -b v1.0-benchmarked https://github.com/ethemdenizersoy16/qTCP-project.git qTCP-v1
cd qTCP-v1

conda create -n qtcp-v1 python=3.12 -y
conda activate qtcp-v1
pip install -r requirements.txt          # pinned versions used for the paper runs
```

Run all commands from the repository root. `qtcp_bench.py` imports the local
`sequence/` package directly, so no install step is needed.

```bash
# check the plumbing
python qtcp_bench.py --preflight-only

# gate-noise arms (realistic, gate sweep): reproduces benchmark_results/trials_gate_arms.csv
python qtcp_bench.py --full --gate-arms-only --workers 64

# all five v1.0 arms (ideal, realistic, gate sweep, loss, high-loss mixed)
python qtcp_bench.py --full --workers 64
```

- **Output:** by default the runner writes to
  `/tmp/e2727956/qtcp_bench/trials_<tag>.csv`. Use `--out PATH` to change this.
- **Resuming:** runs can be resumed. If no trials are left, the runner prints
  the summary (success rate with 95% Clopper–Pearson intervals, mean fidelity,
  send time).
- **Determinism:** seeds are `(MASTER_SEED << 20) + job_index` with
  `MASTER_SEED = 20260817`, so reruns reproduce rows exactly.

## Data

| File | Contents |
|---|---|
| `benchmark_results/trials_gate_arms.csv` (this branch) | v1.0 realistic and gate-sweep arms, 26 cells × (2000 random-state + 399 fixed-state trials). **Paper source** for all v1.0 results with F_gate < 1. |
| `benchmark_results/trials_full.v1.csv` (on `main`) | v1.0 ideal, loss and high-loss mixed arms. **Paper source** for those arms. Its `realistic` and `gatesweep` rows use the old gate-noise injector and are superseded by the file above. |

The gate-arm rows use the corrected gate-noise injector committed in `631cae7`.
It draws a Pauli error with probability ((d+1)/d)(1 − F_gate) on each gate's own
operands and is identical to the injector on `main`. The rows are stamped
`a022e8a-dirty` because the run used that script on top of `a022e8a` before it
was committed. Its success rates, mean fidelities and send times match the
original September run.

The CSV columns and the success / silent-error definitions are the same as on
`main`. `delivered = 1` means the outcome is `DELIVERED` and fidelity > 0.999.
See the `main` README for the full column table.

## Acknowledgements

Built on [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe)
(Argonne National Laboratory).

## License

BSD 3-Clause-style license inherited from SeQUeNCe; see [`LICENSE`](LICENSE).