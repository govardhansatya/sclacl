#!/usr/bin/env python3
"""
PRICE vs SC-LACL regime-stress comparison
==========================================
Companion to sclacl_experiments.py. Tests whether SC-LACL's adaptive
Lyapunov queues (Q_E, Q_Z) earn their keep over PRICE -- a static
shadow-price baseline that uses the identical top-k wear-leveled update
mechanism but with Q_E/Q_Z replaced by constants calibrated once (from
SC-LACL's own long-run average on ELEC2) and then frozen.

Finding used in the paper (Sec. "Why it works" / PRICE paragraph):
  - Table 1's regime and moderately tight endurance (E_cyc=2.5e3): PRICE
    and SC-LACL are numerically identical -- the frozen price never needs
    to move, so adaptivity adds nothing there.
  - Severe, out-of-calibration scarcity (E_cyc=200, V=2 for both): PRICE
    keeps higher accuracy (0.795 vs 0.756) but needs 4.6x more devices
    (23 vs 5) and a 4.4x worse tCLDP (28.97 vs 6.62), because its
    calibrated wear price is zero, so it never throttles writes.

Run from the repository root: python3 code/price_regime_stress.py
Writes results/exp_price_regimes.csv.
"""
import copy, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository root
from sclacl_experiments import Cfg, run, calibrate_price


def summarize(dataset, cfg, seeds=5):
    out = {}
    for pol in ["NAIVE", "PRICE", "SCLACL"]:
        rs = [run(dataset, pol, cfg, sd) for sd in range(seeds)]
        out[pol] = dict(
            acc=np.mean([r["mean_acc"] for r in rs]),
            acc_std=np.std([r["mean_acc"] for r in rs]),
            dev=np.mean([r["devices"] for r in rs]),
            Mops=np.mean([r["Mops"] for r in rs]),
            blackouts=np.mean([r["blackouts"] for r in rs]),
            tCLDP=np.mean([r["tCLDP"] for r in rs]),
            tCLDP_std=np.std([r["tCLDP"] for r in rs]),
        )
    return out


def main():
    base = Cfg()
    price_w, price_e = calibrate_price(base)
    print(f"PRICE calibrated once on ELEC2/default regime: price_w={price_w:.4f} price_e={price_e:.4f}\n")

    regimes = {
        "main regime (Table 1 defaults)": dict(),
        "tight endurance (E_cyc=2500)": dict(endurance_cycles=2.5e3),
        "scarce harvest (mean=0.45)": dict(harvest_mean=0.45),
        "extreme scarcity (E_cyc=200, V=2)": dict(endurance_cycles=200.0, V=2.0),
    }
    rows = ["regime,policy,acc,acc_std,dev,Mops,blackouts,tCLDP,tCLDP_std"]
    for name, overrides in regimes.items():
        cfg = copy.deepcopy(base)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.price_w, cfg.price_e = price_w, price_e
        print(f"=== {name} on ELEC2 ===")
        for pol, v in summarize("ELEC2", cfg).items():
            print(f"  {pol:8s} acc={v['acc']:.4f}±{v['acc_std']:.4f}  dev={v['dev']:.1f}  "
                  f"Mops={v['Mops']:.3f}  blackouts={v['blackouts']:.1f}  "
                  f"tCLDP={v['tCLDP']:.3f}±{v['tCLDP_std']:.3f}")
            rows.append(f"{name.replace(',', '')},{pol},{v['acc']:.4f},{v['acc_std']:.4f},{v['dev']:.1f},"
                        f"{v['Mops']:.3f},{v['blackouts']:.1f},{v['tCLDP']:.3f},{v['tCLDP_std']:.3f}")
        print()
    (ROOT / "results" / "exp_price_regimes.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
