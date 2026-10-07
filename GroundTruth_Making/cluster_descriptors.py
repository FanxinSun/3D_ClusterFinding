#!/usr/bin/env python
"""
cluster_descriptors.py [--out data/labels_v0]

One entry per official TPC cluster with recovered == 1 (the selection that
replaces idmask == 15), computed from its recovered hits, per event. Reads the
per-event hits_evNNN.root / clusters_evNNN.root written by label_tpc_hits.py
and writes data/labels_v0/cluster_descriptors.root (tree desc).

Adapted from an earlier prototype (cluster_features/_wmean), with these
differences:
  - grouping key is the (globally unique) cluster id `cid`/`cl_id`, not
    (event, layer, side, cid) -- no phi-unwrap (moments are computed in
    cluster-LOCAL relative pad/tbin coordinates, see next point), so the
    "phibin_is_global" unwrap block does not apply and is dropped.
  - coordinates are relative to the cluster's own minimum phibin / minimum
    tbin (subtracted per cluster before any moment is computed) -- the raw
    E[phi^k] formulas lose digits at phibin ~ 2000; the weighted mean of a
    per-hit quantity is unchanged by first collapsing hits that share a pad
    (or a tbin) and summing their adc, so this is also how the phi-only /
    t-only marginals are formed (sum over tbin per pad for phi; sum over pads
    per tbin for time).
  - n_sat and p2 (phisize==2 charge-sharing fraction) computed via groupby
    aggregation, no per-group python lambda.
  - one event's hits/clusters are read, described and discarded before the
    next event, keeping memory well under 4 GB regardless of run length.

A cluster is skipped entirely if any of its assigned hits has adc_anomalous
set (adc > 1023) -- those hits are kept in hits_evNNN.root, just not described.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import uproot

from root_io_utils import write_root_tree_fixed

LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels_v0")   # product directory; override on a new machine
ADC_SAT = 963

DESC_SCHEMA = {
    "event": np.int16, "cl_id": np.int64,
    "layer": np.int8, "sector": np.int8, "side": np.int8, "region": np.int8,
    "ontrack": np.uint8, "clean": np.uint8,
    "pedge": np.int32, "phisize": np.int32, "zsize": np.int32,
    "n_hits": np.int32, "n_sat": np.int32,
    "adc_max": np.int32, "adc_sum": np.int32, "adc_min": np.int32,
    "cog_phi": np.float32, "cog_t": np.float32, "rms_phi": np.float32, "rms_t": np.float32,
    "skew_phi": np.float32, "skew_t": np.float32, "kurt_phi": np.float32, "kurt_t": np.float32,
    "corr_phi_t": np.float32, "in_drift_window": np.uint8, "p2": np.float32,
    "tbin_abs": np.float32,     # official cluster's own stored tbin -- an absolute drift-time
                                 # reference; cog_t is cluster-LOCAL (relative to the cluster's own
                                 # minimum tbin, to avoid precision loss) and is not usable for
                                 # a global "is this cluster in the drift window" check or for
                                 # plotting a quantity against absolute drift position (rms_t vs tbin).
    "cog_phi_abs": np.float32,  # cluster's minimum phibin + cog_phi (local)
    "cog_t_abs": np.float32,    # cluster's minimum tbin + cog_t (local)
}


def region(layer):
    return np.where(layer < 23, 1, np.where(layer < 39, 2, 3))


def _wmean(w, x, key):
    return (w * x).groupby(key).sum()


def compute_descriptors(h, c, event):
    anomalous_cids = set(h.loc[h.adc_anomalous == 1, "cid"].unique()) - {-1}   # clusters holding an anomalous-adc hit are skipped
    sel_c = c[(c.recovered == 1) & (~c.cl_id.isin(anomalous_cids))].copy()
    if not len(sel_c):
        return pd.DataFrame(columns=list(DESC_SCHEMA))
    keep_cids = set(sel_c.cl_id.values)
    hh = h[h.cid.isin(keep_cids)].copy()
    if not len(hh):
        return pd.DataFrame(columns=list(DESC_SCHEMA))

    key = hh["cid"]
    pmin = hh.groupby(key)["phibin"].transform("min")
    tmin = hh.groupby(key)["tbin"].transform("min")
    p = (hh["phibin"] - pmin).astype(np.float64)
    t = (hh["tbin"] - tmin).astype(np.float64)
    w = hh["adc"].astype(np.float64)
    Wser = w.groupby(key).sum()

    mom = {}
    for name, x in (("phi", p), ("t", t)):
        E = {k: _wmean(w, x ** k, key) / Wser for k in (1, 2, 3, 4)}
        mu, m2 = E[1], (E[2] - E[1] ** 2).clip(lower=0)
        m3 = E[3] - 3 * mu * E[2] + 2 * mu ** 3
        m4 = E[4] - 4 * mu * E[3] + 6 * mu ** 2 * E[2] - 3 * mu ** 4
        mom[f"cog_{name}"] = mu
        mom[f"rms_{name}"] = np.sqrt(m2)
        mom[f"skew_{name}"] = np.where(m2 > 0, m3 / m2 ** 1.5, np.nan)
        mom[f"kurt_{name}"] = np.where(m2 > 0, m4 / m2 ** 2 - 3, np.nan)   # excess kurtosis

    Ept = _wmean(w, p * t, key) / Wser
    cov = Ept - mom["cog_phi"] * mom["cog_t"]
    denom = mom["rms_phi"] * mom["rms_t"]
    corr_phi_t = np.where(denom > 0, cov / denom, np.nan)

    g = hh.groupby(key)
    n_hits = g.size()
    adc_max = g["adc"].max(); adc_sum = g["adc"].sum(); adc_min = g["adc"].min()
    n_sat = (hh["adc"] >= ADC_SAT).astype(np.int64).groupby(key).sum()   # vectorised, no lambda
    pmin_cl = g["phibin"].min(); tmin_cl = g["tbin"].min()   # per-cluster (not per-hit) minimum

    desc = pd.DataFrame({
        "cog_phi": mom["cog_phi"], "cog_t": mom["cog_t"],
        "cog_phi_abs": pmin_cl.astype(np.float64) + mom["cog_phi"],
        "cog_t_abs": tmin_cl.astype(np.float64) + mom["cog_t"],
        "rms_phi": mom["rms_phi"], "rms_t": mom["rms_t"],
        "skew_phi": mom["skew_phi"], "skew_t": mom["skew_t"],
        "kurt_phi": mom["kurt_phi"], "kurt_t": mom["kurt_t"],
        "corr_phi_t": corr_phi_t,
        "n_hits": n_hits, "adc_max": adc_max, "adc_sum": adc_sum, "adc_min": adc_min, "n_sat": n_sat,
    })
    desc.index.name = "cl_id"
    desc = desc.reset_index()

    meta = sel_c.set_index("cl_id")[["layer", "phielem", "zelem", "pedge", "phisize", "zsize",
                                      "ontrack", "clean", "tbin"]].rename(columns={"tbin": "tbin_abs"})
    desc = desc.merge(meta, left_on="cl_id", right_index=True, how="left")
    desc.insert(0, "event", event)
    desc["sector"] = desc["phielem"]; desc["side"] = desc["zelem"]
    desc["region"] = region(desc["layer"].values)
    desc["in_drift_window"] = ((desc["tbin_abs"] >= 0) & (desc["tbin_abs"] <= 251)).astype(np.uint8)

    # p2 (phisize==2 only): charge fraction on the lesser pad, from the pad-only marginal --
    # vectorised (groupby.min/.sum over (cid, relative-pad) pairs), no per-group lambda.
    prof2 = hh.assign(p_rel=p).groupby([key, "p_rel"])["adc"].sum()
    g2 = prof2.groupby(level=0)
    p2_all = (g2.min() / g2.sum()).rename("p2")
    desc = desc.merge(p2_all, left_on="cl_id", right_index=True, how="left")
    desc.loc[desc.phisize != 2, "p2"] = np.nan

    return desc[list(DESC_SCHEMA)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=LABELS_DIR)
    a = ap.parse_args()

    hit_files = sorted(glob.glob(f"{a.out}/hits_ev*.root"))
    all_desc = []
    for hp in hit_files:
        ev = int(hp.split("hits_ev")[-1].split(".root")[0])
        cp = f"{a.out}/clusters_ev{ev:03d}.root"
        h = uproot.open(hp)["hits"].arrays(["cid", "phibin", "tbin", "adc", "adc_anomalous"], library="pd")
        c = uproot.open(cp)["clusters"].arrays(
            ["cl_id", "layer", "phielem", "zelem", "pedge", "phisize", "zsize",
             "ontrack", "clean", "recovered", "tbin"], library="pd")
        d = compute_descriptors(h, c, ev)
        all_desc.append(d)
        print(f"ev {ev:3d}: {len(d)} descriptor entries (recovered clusters)")
        del h, c

    desc_all = pd.concat(all_desc, ignore_index=True) if all_desc else pd.DataFrame(columns=list(DESC_SCHEMA))
    write_root_tree_fixed(f"{a.out}/cluster_descriptors.root", "desc", desc_all, DESC_SCHEMA)
    print(f"\n{len(desc_all)} total descriptor entries -> {a.out}/cluster_descriptors.root")


if __name__ == "__main__":
    main()
