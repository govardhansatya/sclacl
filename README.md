# SC-LACL: Lifetime-Aware Continual Learning on Wear-Limited Edge Hardware

Code, results and figures for the paper

> Gadi Govardhan Satya, Joshitha Nambakam, and Hrudya P. 2026. **The Adaptation–Embodied Carbon
> Paradox: A Lifetime-Aware Theory of Continual Learning on Wear-Limited Edge Hardware.**
> In *Proceedings of the ACM AI Summit 2026 (AI Summit '26)*, Atlanta, GA, USA.
> https://doi.org/10.1145/3806096.3844884

`paper/` holds the proceedings version and an extended version with all proofs.

Continual learning on microcontrollers writes model updates into non-volatile memory (NVM) with
finite program/erase (P/E) endurance. Every update shortens device lifetime, and a replaced device
costs its embodied carbon again. SC-LACL (Stability-Controlled Lifetime-Aware Continual Learning)
controls updates with energy and wear virtual queues, applies top-k sparse, wear-biased updates,
and gates each update with a carbon break-even rule. The paper evaluates it with the Total
Carbon-Lifetime-Delay Product (tCLDP).

## Repository layout

```
code/
  sclacl_experiments.py   main harness: 8 policies x 3 streams x 5 seeds, plus ablations
  price_regime_stress.py  PRICE vs SC-LACL vs Naive in four regimes (ELEC2)
  make_split_figs.py      the five paper figures
  plot_results.py         a combined three-panel summary figure
  fetch_data.py           downloads ELEC2 and Covertype and checks their SHA-256
data/                     datasets (downloaded, not committed; see data/README.md)
results/                  CSV outputs used in the paper
results_preprint/         outputs used for the earlier preprint, before the energy fix
figures/                  the five figures in the paper
paper/                    proceedings PDF and extended version PDF
```

## Setup

Python 3.10 or newer, CPU only.

```bash
pip install -r requirements.txt     # numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9
python code/fetch_data.py           # ELEC2 + Covertype into data/
```

## Reproducing the paper

| Command | Output | Used for |
|---|---|---|
| `python code/sclacl_experiments.py` | `results/exp_main.csv`, `results/exp_ablation.csv`, `results/exp_config.json` | Table 1, Table 2, Figs. 3 to 5 |
| `python code/price_regime_stress.py` | `results/exp_price_regimes.csv` | PRICE paragraph and regime analysis |
| `python code/make_split_figs.py` | `figures/sclacl_*_v2.png` | Figs. 1 to 5 |
| `python code/plot_results.py` | `figures/sclacl_exp_results.png` | summary figure (not in the paper) |

All scripts locate `data/`, `results/` and `figures/` relative to the repository root. Seeds are
fixed (0 to 4), so with the package versions above a re-run reproduces `results/` exactly and the
figures pixel for pixel.

## Main settings

| Setting | Value |
|---|---|
| Streams | ELEC2 (45,312 samples), Covertype (40,000-sample block), SEA (40,000 samples, generated) |
| Time model | one sample per 30 min; 48 samples per day; horizon 2.59 yr (ELEC2), 2.28 yr (Covertype, SEA) |
| Model | one-hidden-layer MLP (64 units), learning rate 0.05, minibatch 16 |
| NVM | endurance E_cyc = 10^4 P/E per cell; every written weight counts one P/E on its cell |
| Embodied carbon | 2.59 kg CO2e per device; a device is replaced when its most-written cell reaches E_cyc |
| Energy | seasonal solar harvest (mean 0.95 per day), supercapacitor buffer, write energy per operation |
| Baselines | Static, Naive, Periodic, EWC, Experience Replay, LwF: 6 epochs per day |
| SC-LACL | V = 12, top-k = 0.18, wear budget 0.85 E_cyc / T, 1 to 4 epochs per day scaled by drift |
| PRICE | SC-LACL's update with the queues replaced by constants calibrated once on ELEC2 |

## Headline results (mean over 5 seeds, `results/exp_main.csv`)

| Stream | Method | Accuracy | Devices | Writes (M P/E) | tCLDP |
|---|---|---|---|---|---|
| ELEC2 | Naive CL | 0.802 | 2 | 17.1 | 2.498 |
| ELEC2 | LwF (best accuracy) | 0.809 | 2 | 17.1 | 2.475 |
| ELEC2 | SC-LACL | 0.795 | 1 | 0.84 | **1.260** |
| Covertype | Naive CL | 0.860 | 1 | 30.4 | 1.319 |
| Covertype | SC-LACL | **0.867** | 1 | 2.3 | **1.310** |
| SEA | LwF (best accuracy) | 0.871 | 2 | 5.65 | 2.607 |
| SEA | SC-LACL | 0.865 | 1 | 0.19 | **1.312** |

SC-LACL issues 13 to 30 times fewer writes than the wear-agnostic baselines. On ELEC2 and SEA the
baselines exhaust memory at 1.6 years and need a second device; on Covertype they are throttled by
harvested energy instead.

## Notes for interpreting the results

- These are simulations. Accuracy is measured on real drift streams, and wear is counted from the
  optimizer's actual per-cell writes under datasheet endurance and energy constants. They are not
  hardware lifetime measurements.
- tCLDP is computed per seed as devices x 2.59 kg / horizon / mean accuracy, then averaged. With a
  high-variance method (EWC on Covertype) this differs from the value computed from mean accuracy.
- `MTTF` in `exp_main.csv` is the time of the first endurance failure; it equals the horizon when no
  device fails.
- The wear queue tracks mean wear per cell. Bias tensors with 8 or fewer entries are always updated
  densely (top-k and wear-biasing only apply to larger tensors), so under tight endurance
  (E_cyc = 2,500) these cells wear out first and every SC-LACL variant needs two devices.
- PRICE's calibrated wear price is 0 and its energy price is 0.0033, because SC-LACL's wear queue
  stays empty on ELEC2 in the default regime. PRICE therefore never penalizes wear, and it matches
  SC-LACL wherever the wear gate is idle. In the extreme-scarcity regime (E_cyc = 200) both use V = 2.
- Write reduction on ELEC2 (17.1 M to 0.84 M, about 20 times) splits into about 5.5 times from top-k
  sparsity and about 3.7 times from the drift-scaled epoch schedule. In the default regime the
  break-even gate and wear-biasing are inactive (Table 2).
- EWC uses a simple Fisher estimate (squared batch gradient, decay 0.9) and is unstable on some seeds
  (ELEC2 seed 4, Covertype seed 0). Results are reported as run.
- The near-optimality theorem adapts the standard drift-plus-penalty result (Neely, 2010) to an
  idealized controller; the simulated controller estimates the loss change from recent accuracy.

## Energy-accounting fix

Before the camera-ready version, a day's training ran before its energy was checked, so writes the
buffer and harvest could not pay for still happened and only inference energy was charged. Every
policy now trains only as many epochs as it can pay for. This changed only runs where energy was
short: the dense baselines on Covertype (now energy-throttled) and the scarce-harvest regime
(Naive: 126 brownout days, accuracy 0.695). ELEC2, SEA, the ablations and the other regimes are
unchanged. `results_preprint/` keeps the earlier outputs used for the Zenodo preprint
(doi:10.5281/zenodo.22111193).

## Citation

```bibtex
@inproceedings{satya2026paradox,
  author    = {Gadi Govardhan Satya and Joshitha Nambakam and Hrudya P},
  title     = {The Adaptation--Embodied Carbon Paradox: A Lifetime-Aware Theory of Continual
               Learning on Wear-Limited Edge Hardware},
  booktitle = {Proceedings of the ACM AI Summit 2026 (AI Summit '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Atlanta, GA, USA},
  doi       = {10.1145/3806096.3844884}
}
```

## License

Code (`code/`): MIT, see `LICENSE`. Paper, figures and results: CC BY 4.0, see
`LICENSE-CC-BY-4.0.md`. Datasets keep their original licenses (see `data/README.md`).
