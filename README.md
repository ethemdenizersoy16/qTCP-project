# qTCP: Benchmarking and Dynamic Decoding

A quantitative benchmark of the qTCP quantum transport protocol
[[Yu, Lai & Zhou, IEEE TQE 2021]](https://arxiv.org/abs/1903.10685),
implemented on the [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe)
quantum network simulator, together with a novel dynamic-decoding variant
(v2.0) that resolves the original protocol's poor performance under loss.

## Contributions

This repository contains:

1. **The first implementation of qTCP** as specified in Yu, Lai & Zhou 2021.
   The original paper describes the protocol abstractly; concrete design
   decisions (QPing usage, restart handshakes, fail-stop policy, memory
   allocation) are documented in the accompanying report.

2. **A quantitative benchmark** of qTCP against bare teleportation across
   four failure regimes (entanglement-generation noise, gate infidelity,
   per-transfer loss, and mixed loss+corruption), at N=2000 trials per
   parameter point with Clopper-Pearson intervals.

3. **A dynamic-decoding variant (v2.0)** that switches per-packet between
   correction mode (all shares arrive → syndrome extraction) and erasure
   mode (some shares lost → reconstruct from survivors), exploiting the
   Cleve-Gottesman-Lo equivalence between ((3,5)) QSS and [[5,1,3]] QEC.

4. **A comparison with QARQ** [[Iqbal et al., Sensors 2023]](https://doi.org/10.3390/s23187891),
   the closest published protocol. QARQ and qTCP-v2.0 occupy complementary
   points on the (entanglement cost, delivered fidelity) frontier.

## Headline results

At 30% per-transfer loss and near-perfect entanglement fidelity:

| protocol           | delivery rate | pairs per delivery |
|--------------------|---------------|--------------------|
| bare teleportation | 0.70          | 1                  |
| qTCP baseline      | 0.04          | ~15                |
| qTCP-v2.0          | 0.99          | ~24                |

The baseline confirms the negative result flagged by Iqbal et al.:
qTCP-as-specified is worse than bare teleportation across every non-trivial
operating point. qTCP-v2.0 addresses this: break-even against bare shifts
from ℓ=0.036 (baseline) to past ℓ=0.60 (v2.0) on pure loss, and from
ε=0.036 to ε=0.135 on pure corruption.

## Repository layout

This repository is a fork of SeQUeNCe with qTCP added as a new application
under `sequence/app/qtcp/`, plus a benchmark harness at the repo root.
Everything else is upstream SeQUeNCe unchanged.

```
├── sequence/app/qtcp/          # protocol implementation
│   ├── qtcp_overseer.py        # packet lifecycle, encoding, decoding
│   ├── qtcp_transfer.py        # single-qubit teleportation layer
│   ├── qtcp_handshake.py       # connection setup, QPing quality test
│   ├── qec.py                  # [[5,1,3]] stabilizer code
│   └── qss.py                  # ((3,5)) quantum secret sharing
├── qtcp_bench.py               # benchmark runner (multiprocess, resumable)
├── tmp/three_node.json         # topology config (Alice, Bob, Charlie)
└── benchmark_results/          # raw benchmark CSVs
```

## Branches

- `main` — qTCP-v2.0 with dynamic decoding (current work).
- `v1.0-benchmarked` — qTCP baseline, tagged at the commit that produced
  the reference baseline CSV. Use this branch to reproduce the negative
  result.

## Reproducing the results

Requires Python 3.12 and a Linux environment.

```bash
# clone and install in editable mode (following SeQUeNCe's install convention)
git clone https://github.com/ethemdenizersoy16/qTCP-project.git
cd qTCP-project
make install_editable

# verify plumbing (~30 seconds)
python qtcp_bench.py --preflight-only

# small pilot to confirm behaviour (~5 minutes)
python qtcp_bench.py --pilot

# full benchmark (~30 hours on 64 cores)
python qtcp_bench.py --full --workers 64
```

Results write incrementally to CSV and are resumable — a killed run picks
up where it left off. Full-run parameters (grids, trial counts, seeds)
are at the top of `qtcp_bench.py`.

## Data

Raw benchmark CSVs for both baseline (v1.0) and dynamic-decoding (v2.0)
sweeps will be added to `benchmark_results/` once all runs complete.
Each row represents one full packet transmission with all metrics
recorded (delivery, fidelity, entanglement pairs consumed, gate count,
send time, seed for reproduction).

## Citation

If this work informs your research, please cite:

```
Ersoy, E. D. (2026). Benchmarking qTCP: negative results, dynamic
decoding, and a comparison with QARQ.
```

## Acknowledgements

Built on [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe)
(Argonne National Laboratory). Developed as part of an undergraduate
research internship at WINSLAB.

## License

MIT (matching SeQUeNCe's license).