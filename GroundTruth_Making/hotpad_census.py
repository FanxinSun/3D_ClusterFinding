#!/usr/bin/env python
"""
hotpad_census.py [--out data/labels_v0]

Hits per (layer, zelem, phibin) from all events except 44 (laser);
robust z = (n - median) / (1.4826 * MAD) computed within each (layer, zelem).
Reports per region: number of pads with z > 5, their share of all hits, the 20
highest-z pads, and the median adc on those pads vs all pads.
Writes data/labels_v0/hotpad_census.csv and appends ledger lines.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import uproot

LASER_EVENT = 44
LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels_v0")   # product directory; override on a new machine


def region(layer):
    return np.where(layer < 23, 1, np.where(layer < 39, 2, 3))


def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    denom = 1.4826 * mad
    if denom <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - med) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=LABELS_DIR)
    a = ap.parse_args()

    parts = []
    for hp in sorted(glob.glob(f"{a.out}/hits_ev*.root")):
        ev = int(hp.split("hits_ev")[-1].split(".root")[0])
        if ev == LASER_EVENT:
            continue
        parts.append(uproot.open(hp)["hits"].arrays(["layer", "zelem", "phibin", "adc"], library="pd"))
    allh = pd.concat(parts, ignore_index=True)
    del parts
    print(f"hits loaded (excl. event {LASER_EVENT}): {len(allh)}")

    g = allh.groupby(["layer", "zelem", "phibin"])
    pad = pd.concat([g.size().rename("n"), g["adc"].median().rename("pad_adc_median")], axis=1).reset_index()
    pad["z"] = pad.groupby(["layer", "zelem"])["n"].transform(robust_z)
    pad["region"] = region(pad["layer"].values)
    pad["is_hot"] = pad["z"] > 5

    pad.to_csv(f"{a.out}/hotpad_census.csv", index=False)

    allh2 = allh.merge(pad[["layer", "zelem", "phibin", "region", "is_hot"]],
                        on=["layer", "zelem", "phibin"], how="left")

    ledger_lines = [f"\n=== Phase 2c: hot-pad census (excl. event {LASER_EVENT}) ==="]
    for reg in (1, 2, 3):
        subh = allh2[allh2.region == reg]
        subpad = pad[pad.region == reg]
        total_hits = len(subh)
        hot_hits = subh[subh.is_hot]
        n_hot_pads = int(subpad.is_hot.sum())
        share = float(len(hot_hits) / total_hits) if total_hits else float("nan")
        median_adc_hot = float(hot_hits["adc"].median()) if len(hot_hits) else float("nan")
        median_adc_all = float(subh["adc"].median()) if total_hits else float("nan")
        line = (f"region {reg}: n_pads_z>5={n_hot_pads}  share_of_hits={share:.6f}  "
                f"median_adc(hot pads)={median_adc_hot:.1f}  median_adc(all pads)={median_adc_all:.1f}")
        print(line)
        ledger_lines.append(line)
        top20 = subpad.sort_values("z", ascending=False).head(20)
        ledger_lines.append(f"  top 20 by z (layer, zelem, phibin, n, z, pad_adc_median):")
        for _, r in top20.iterrows():
            ledger_lines.append(f"    L{int(r.layer)} side{int(r.zelem)} phibin{int(r.phibin)}  "
                                 f"n={int(r.n)} z={r.z:.2f} adc_median={r.pad_adc_median:.1f}")

    with open(f"{a.out}/RESULTS_LEDGER.txt", "a") as fh:
        fh.write("\n".join(ledger_lines) + "\n")
    print(f"\nhotpad_census.csv written ({len(pad)} pads); ledger updated.")


if __name__ == "__main__":
    main()
