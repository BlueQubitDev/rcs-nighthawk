# Severing-attack analysis (Appendix F)

Measures what a severing (patch-product) spoofer can actually score against
the experimental circuits, and where the experiment sits relative to the
noise-induced transition.

The attack: partition the qubits into `(A, Ā)`, delete the CZ gates crossing
the cut, sample the two sides independently, submit the product samples. The
score such samples obtain against the *uncut* circuit is computed here
exactly — as an inner product of the target probability vector with the
severed product distribution, with no sampling noise — on sub-regions small
enough to simulate, and transferred to full size using the measured
boundary-locality of the decay.

## Scripts

| script | what it does | runtime |
|---|---|---|
| `min_cut.py` | randomized min-cut search over the coupler graph: best cut vs. side size, severed CZ per cycle, degree-2 modes | ~1 min |
| `spoof_attack.py` | exact attack score of single circuit instances on an m-qubit region, for balanced / min-cut / pendant cuts; fits the per-severed-CZ retained fraction `c` | minutes |
| `spoof_ensemble.py` | the same scores averaged over many independently drawn ensemble instances, which is what the deep-circuit means require | ~40 min per region |

Single-instance scores of a small-`|A|` mode fluctuate by `~2^{-(m-|A|)/2}`,
which buries the mean at large depth; `spoof_ensemble.py` is the version used
for the numbers quoted in the paper.

## Reproducing Appendix F

```bash
python spoofing/min_cut.py --circuit data/circuits/full/d36_logical.qpy \
    --out spoofing/results/min_cuts.json

python spoofing/spoof_ensemble.py --circuit data/circuits/full/d36_logical.qpy \
    --region-center 5  --region-size 20 --arm "[5,6,14]" --instances 120 \
    --out spoofing/results/ensemble_arm3.json
python spoofing/spoof_ensemble.py --circuit data/circuits/full/d36_logical.qpy \
    --region-center 45 --region-size 20 --instances 120 \
    --out spoofing/results/ensemble_c45.json
python spoofing/spoof_ensemble.py --circuit data/circuits/full/d36_logical.qpy \
    --region-center 31 --region-size 20 --instances 120 \
    --out spoofing/results/ensemble_c31.json
```

## Results (61 qubits, 102 couplers, 36 cycles)

The graph admits a balanced `27+34` cut through 6 couplers, an 11-qubit block
through 5, a 4-qubit block and a 3-qubit arm through 3, and eight degree-2
qubits. Against the 36-cycle target `F_XEB = 2.2e-3`:

| branch | b | χ(36) | margin |
|---|---|---|---|
| balanced 27+34 | 6 | 1e-17 – 1e-9 (extrapolated) | >1e6 |
| 11-qubit block | 5 | 1e-14 – 1e-7 (extrapolated) | >1e4 |
| pendant arms, two-sided | 2–3 | 2.2(2.0)e-4 | 10 |
| adjacent pairs, one-sided | 2–4 | <1.5e-4 | 15 |
| singletons, one-sided | 2–3 | <1.3e-4 | 17 |

No branch reaches the experimental score, so the family is excluded on score
alone, without any comparison of what it would cost to run. The measured
retained fraction is `c = 0.48–0.64` per severed CZ with prefactors
`K = 3–50`. `spoof_ensemble.py` also reports the Porter–Thomas collision
ratio, which reaches 1.00 by 32 cycles in every region probed
(anticoncentration).

The three regions cover five of the eight degree-2 qubits; the remaining
three lie outside them and are covered only by the rate argument.
