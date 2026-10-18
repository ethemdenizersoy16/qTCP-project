# qTCP: Implementation, Benchmarking, and Dynamic Decoding

Code and raw data for *"qTCP: Implementation, Benchmarking, and Dynamic Decoding"*
by Ethem Deniz Ersoy and Ertan Onur (WINS Lab, Middle East Technical University),
submitted to IEEE ICC 2027.

qTCP [[Yu, Lai & Zhou, IEEE TQE 2021]](https://arxiv.org/abs/1903.10685) is a
quantum transport protocol that layers ((3,5)) quantum secret sharing (QSS) over
the [[5,1,3]] quantum error-correcting code (QEC) to tolerate loss and errors.
It was proposed abstractly. This repository implements it on the
[SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe) quantum network
simulator, benchmarks it against bare teleportation, and adds a dynamic-decoding
variant (v2.0) that fixes the original protocol's poor performance under loss.

## Contributions

1. **An implementation of qTCP** (v1.0, branch `v1.0-benchmarked`). To our
   knowledge there was no prior public implementation. The original paper leaves
   concrete design decisions open (QPing usage, restart handshakes, fail-stop
   policy, memory allocation). The paper documents the choices made here.
2. **A quantitative benchmark** of qTCP against bare teleportation across four
   failure regimes: entanglement-generation noise, gate infidelity,
   per-transfer loss, and mixed loss plus corruption. It uses 2000 random-state
   trials per parameter cell (≈378,000 in total) with Clopper–Pearson intervals.
3. **Dynamic decoding** (v2.0, this branch). It switches per block between
   correction decoding (all five shares arrive) and erasure decoding (3–4 shares
   arrive). It relies on the equivalence between ((3,5)) QSS and [[5,1,3]] QEC.
4. **A comparison with QARQ**
   [[Iqbal et al., Sensors 2023]](https://doi.org/10.3390/s23187891).
   qTCP-v2.0 and QARQ sit at different points on the (entanglement cost,
   delivered fidelity) frontier, and neither dominates.

## Headline results

All results are at tree depth one.

|                                                        | Bare        | v1.0      | v2.0          |
|--------------------------------------------------------|-------------|-----------|---------------|
| Delivery at ℓ = 0.30 (perfect gates and entanglement)  | 0.70        | 0.04      | 0.99          |
| Entangled pairs per attempt                            | 1           | 14.5      | 24.4          |
| Break-even vs. bare, pure loss                         | –           | ℓ ≈ 0.032 | ℓ > 0.60      |
| Break-even vs. bare, pure corruption (ideal gates)     | –           | ε ≈ 0.034 | ε ≈ 0.14–0.18 |
| Delivery, F_gate = 0.999, F_ent = 0.97, ℓ = 0.10 / 0.30 | 0.87 / 0.68 | not run   | 0.92 / 0.80   |

qTCP as specified (v1.0) is worse than bare teleportation at almost every
non-trivial operating point tested. That confirms the qualitative critique by
Iqbal et al. Dynamic decoding reverses this. The trade-off is silent
miscorrection in the erasure branch. Silent errors are counted as failures in
every reported success rate.

## How v1.0 and v2.0 differ

Both versions use the same encoding pipeline. Each logical qubit is QSS-encoded
into five shares, and each share is protected by [[5,1,3]] QEC (the "leaves").
If held plus delivered shares of a non-leaf block drop to k = 3, Alice
recursively QSS-encodes its next share. Leaves are never recursed on.

- **v1.0 (fail-stop):** a leaf goes LOST on its first failed share. QSS layers
  stop sending once k = 3 shares are delivered, and Bob always erasure-decodes
  from three shares.
- **v2.0 (dynamic decoding):** all n = 5 shares are sent until a failure occurs,
  then sending stops at k. Bob picks the decoder per block by arrival count:
  5 arrivals means QEC correction, and 3–4 arrivals means erasure decoding.
  Leaves get one restart.

## Repository layout

This repository is a fork of SeQUeNCe. qTCP is added as an application under
`sequence/app/qtcp/`, with a benchmark harness at the repository root.
Everything else is upstream SeQUeNCe.

```
├── sequence/app/qtcp/
│   ├── qtcp_overseer.py        # block lifecycle, dispatch policy, encoding, decoding
│   ├── qtcp_transfer.py        # single-qubit teleportation layer, RTO and retransmission
│   ├── qtcp_handshake.py       # connection setup
│   ├── qping.py                # QPing link-quality test
│   ├── qtcp_app.py             # application wrapper (Alice / Bob)
│   ├── qec.py                  # [[5,1,3]] stabilizer code
│   └── qss.py                  # ((3,5)) quantum secret sharing
├── sequence/topology/qtcp_net_topo.py   # topology used by the benchmark
├── qtcp_bench.py               # benchmark runner (multiprocess, resumable)
├── tmp/three_node.json         # topology config (Alice, Bob, Charlie)
└── benchmark_results/          # raw benchmark CSVs
```

## Branches

- `main`: qTCP-v2.0 with dynamic decoding.
- `v1.0-benchmarked`: the qTCP baseline (v1.0), with its gate-noise raw data.
  Use this branch to reproduce the v1.0 results.

## Reproducing the results

The runs use Python 3.12 on Linux.

```bash
git clone https://github.com/ethemdenizersoy16/qTCP-project.git
cd qTCP-project

conda create -n qtcp python=3.12 -y      # any Python 3.12 environment works
conda activate qtcp
pip install -r requirements.txt          # pinned versions used for the paper runs
make install_editable                    # installs this SeQUeNCe fork in editable mode
```

```bash
# check the plumbing (~30 s)
python qtcp_bench.py --preflight-only

# small pilot to confirm behaviour (~5 min)
python qtcp_bench.py --pilot

# full benchmark, all seven arms
python qtcp_bench.py --full --workers 64

# only the arms with F_gate < 1: realistic, gate sweep, realistic mixed
python qtcp_bench.py --full --gate-arms-only --workers 64

# only the high-corruption mixed arm
python qtcp_bench.py --full --mixed_corr --workers 64
```

- **Output:** by default the runner writes to
  `/tmp/e2727956/qtcp_bench/trials_<tag>.csv`. Use `--out PATH` to change this.
- **Resuming:** rows are written as they finish. Rerun the same command with the
  same `--out` and completed trials are skipped. If nothing is left to run, the
  runner prints the summary table (success rates with 95% Clopper–Pearson
  intervals, mean fidelity, send time, and the silent-error breakdown). This
  also works on the shipped data, for example
  `python qtcp_bench.py --full --gate-arms-only --workers 1 --out benchmark_results/trials_gate_arms.csv`.
- **Determinism:** every trial's seed is `(MASTER_SEED << 20) + job_index`, with
  `MASTER_SEED = 20260817`. A rerun reproduces rows exactly.
- **Configuration:** grids, trial counts (2000 random-state plus 400 fixed-state
  trials per cell) and seeds are at the top of `qtcp_bench.py`.

## Data

Each CSV row is one full block transmission.

| Column | Meaning |
|---|---|
| `arm` | `ideal`, `realistic`, `gatesweep`, `losssweep`, `mixed` (high-loss), `mixed_corr` (high-corruption), `realistic_mixed` |
| `ent_fid`, `gate_fid`, `meas_fid`, `loss_rate` | the parameter cell |
| `state_kind` | `random` (Haar-random input; all paper statistics use these rows) or `fixed_0` / `fixed_plus` / `fixed_iplus` (sanity states) |
| `outcome` | protocol verdict: `DELIVERED`, `LOST`, `RECOVERED_LOCALLY` |
| `delivered` | 1 if `outcome == DELIVERED` and fidelity > 0.999, which is the success metric |
| `fidelity` | fidelity of Bob's state to Alice's input |
| `ent_pairs_consumed`, `gate_count`, `classical_msgs`, `send_time_us`, … | cost metrics |
| `seed_base`, `code_version` | seed for exact replay, and the git hash that produced the row (`-dirty` = uncommitted changes) |

A **silent error** is a row with `outcome == DELIVERED` and `delivered == 0`:
Bob accepted a wrong state. A **detected failure** is any other outcome. In
every cell, success + silent + detected = 1.

### Which file backs which result

| File | Contents | Status |
|---|---|---|
| `trials_full.csv` | v2.0, all seven arms | **Paper source** for ideal, loss, high-loss mixed (`baca556`) and high-corruption mixed (`f55c56d`). Its `realistic`, `gatesweep` and `realistic_mixed` rows use the old gate-noise injector and are superseded. |
| `trials_gate_arms.csv` | v2.0 realistic, gate sweep, realistic mixed | **Paper source** for every v2.0 result with F_gate < 1 (fixed injector). |
| `trials_full.v1.csv` | v1.0 ideal, realistic, gate sweep, loss, high-loss mixed (`6cd0f41-dirty`) | **Paper source** for v1.0 ideal, loss and high-loss mixed. Its `realistic` and `gatesweep` rows are superseded by `benchmark_results/trials_gate_arms.csv` on the `v1.0-benchmarked` branch. |
| `trials_full.v2.csv` | v2.0, five arms (`baca556`) | Same rows as the corresponding subset of `trials_full.csv`. Kept for reference. |
| `trials_full.old.csv` | early partial run (`6cd0f41`) | Superseded. |
| `trials_realistic_only.csv` | preliminary narrow run at F_ent = 0.97 with different seeds | Not used in the paper. |

### Gate-noise provenance

The gate-noise injector draws a Pauli error with probability
((d+1)/d)(1 − F_gate) on each gate's own operands, where d depends on the gate's
arity. It was committed in `6dda5a8`. Arms with F_gate = 1 never invoke it, so
their earlier rows stay valid.

In `trials_gate_arms.csv`, rows stamped `e07efab-dirty` were produced by that
injector before it was committed. Rows stamped `af067c6` complete the same run
after an interruption. The only commits in between touch documentation and
requirements, and the deterministic seeding regenerates identical rows. The
summary therefore prints a "MIXED CODE VERSIONS" warning for this file, which is
expected.

A few rows have no received state. These are detected losses or trials that
recovered locally, not runner errors.

## Acknowledgements

Built on [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe)
(Argonne National Laboratory).

## License

BSD 3-Clause-style license inherited from SeQUeNCe; see [`LICENSE`](LICENSE).