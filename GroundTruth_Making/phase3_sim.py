#!/usr/bin/env python
"""
phase3_sim.py

Comparison of one simulated event with the real-data labelling. One sim event
(v6.1 exports, lowest-numbered = event 0), hits from hit69, clusters+truth from
island91, no track tree (ontrack = 0 always throughout). The printed report,
which is also appended to the labels ledger, has these parts:

3c-1  integer-ness of sim hit adc / cluster adc / cluster maxadc (fact only).
3c-2  PURE-GEOMETRY association: each cluster -> the 4-connected island of the
      nearest hit inside its widened box (no adc matching at all). toff/poff
      come from the calibration run (label_hitset_sim, below): if the median
      dt/dp of the bit-1-passing clusters exceeds 0.2 bin the run is repeated
      once with those offsets; falls back to 0/0 if too few clusters pass bit 1
      to give a stable median (reported either way).
3c-3  For islands the 3c-2 association makes unambiguous (exactly one
      cluster): R_sum, R_max, D_max, bounding-box-size differences, all hits
      of the island vs the official cluster. Same four quantities for REAL
      event 1, there using the existing (recovered-membership) comp/hs/
      comp_ncl already on disk -- no re-derivation needed.
3c-4  Baseline only: 1e-3 relative tolerance on bits 1, 2, the budget test AND
      the anchor match. frac_recovered + idmask histogram, labelled "baseline,
      not tuned".
3d    Side-by-side table: sim event 0, real event 1, real event 25 (the
      busiest complete event of the hitset panels).

No algorithm change beyond what is described above; no cut tuning; numbers
only, no interpretation.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import uproot
from scipy.ndimage import label as cc_label

sys.path.insert(0, os.path.dirname(__file__))
from label_tpc_hits import (build_pad_table, dead_map, add_pad_centroid, region,
                             CENTROID_TOL_T, CENTROID_TOL_PAD, NPAD, NSEC)
from root_io_utils import write_root_tree_fixed

LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels_v0")   # product directory; override on a new machine
TPC_SIM_DIR = os.path.expanduser(os.environ.get("TPC_SIM_DIR", "~/sPHENIX/TPC_Sim_Pipeline/island_post"))
REAL_OUT = LABELS_DIR
REL_TOL = 1e-3
QUANTILES = [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]
MIN_STABLE_N = 30   # bit-1-passing clusters needed to trust the calibration median (stated, not tuned)

SIM_HIT_COLS = ["layer", "phielem", "zelem", "phibin", "tbin", "adc"]
SIM_CL_COLS = ["event", "layer", "phielem", "zelem", "phi", "tbin", "adc", "maxadc",
               "phisize", "zsize", "pedge", "x", "y", "z"]

HITS_SCHEMA_SIM = {
    "event": np.int16, "layer": np.int8, "phielem": np.int8, "zelem": np.int8,
    "phibin": np.int16, "tbin": np.int16, "adc": np.float32, "ientry": np.int32,
    "cid": np.int32, "comp": np.int32, "hs": np.int16,
    "comp_n": np.int32, "comp_ncl": np.int16, "idmask": np.uint8,
    "in_drift_window": np.uint8, "subthr": np.uint8, "edgepad": np.uint8, "deadadj": np.uint8,
    "recovered": np.uint8,
}
CLUSTERS_SCHEMA_SIM = {
    "event": np.int32, "layer": np.int32, "phielem": np.int32, "zelem": np.int32,
    "phi": np.float32, "tbin": np.float32, "adc": np.float32, "maxadc": np.float32,
    "phisize": np.int32, "zsize": np.int32, "pedge": np.int32,
    "x": np.float32, "y": np.float32, "z": np.float32,
    "cl_id": np.int64, "finite": np.uint8, "ix": np.int64, "iy": np.int64, "iz": np.int64,
    "pad_c": np.float32, "idmask": np.uint8, "nhit": np.int32, "dt": np.float32, "dp": np.float32,
    "comp": np.int32, "hs": np.int32, "ncomp": np.int32, "comp_ncl": np.int32,
    "tbin_sentinel": np.uint8, "recovered": np.uint8,
    "cls": np.int8, "gpt": np.float32,
}


def _xyz_keys(d, ok):
    out = []
    for a in ("x", "y", "z"):
        v = d[a].values.astype(np.float64)
        k = np.full(len(v), np.iinfo(np.int64).min, np.int64)
        k[ok] = np.rint(v[ok] * 1e4).astype(np.int64)
        out.append(k)
    return out


def load_sim_clusters(cfile, event):
    c = cfile["ntp_cluster"].arrays(SIM_CL_COLS, library="pd")
    truth = cfile["ntp_truth"].arrays(["cls", "gpt"], library="pd")   # entry-aligned with ntp_cluster (identical event arrays, equal entry counts)
    c["cls"] = truth["cls"].values
    c["gpt"] = truth["gpt"].values
    c = c[(c.event == event) & (c.layer >= 7) & (c.layer <= 54)].reset_index(drop=True)
    c["cl_id"] = np.arange(len(c), dtype=np.int64)
    c["finite"] = np.isfinite(c.x) & np.isfinite(c.y) & np.isfinite(c.z) & np.isfinite(c.tbin) & np.isfinite(c.phi)
    for k in ["layer", "phielem", "zelem", "phisize", "zsize", "pedge"]:
        c[k] = c[k].astype(np.int64)
    c["ix"], c["iy"], c["iz"] = _xyz_keys(c, c.finite.values)
    return c


def read_sim_hits(hfile, event):
    h = hfile["ntp_hit"].arrays(SIM_HIT_COLS + ["event"], library="pd")
    h = h[h.event == event].reset_index(drop=True)
    h["ientry"] = h.index.values.astype(np.int64)
    m = (h.layer >= 7) & (h.layer <= 54) & np.isfinite(h.phibin) & np.isfinite(h.tbin)
    h = h[m].copy()
    for k in ["layer", "phielem", "zelem", "phibin", "tbin"]:
        h[k] = h[k].astype(np.int64)
    return h.reset_index(drop=True)


# ----------------------------------------------------------------------------- baseline recovery run
def label_hitset_sim(P, T, A, c, toff, poff, anchor_rel_tol):
    n, m = len(P), len(c["adc"])
    p0, t0 = P.min(), T.min()
    img = np.zeros((P.max() - p0 + 1, T.max() - t0 + 1), np.int8)
    img[P - p0, T - t0] = 1
    lab, _ = cc_label(img)
    comp = lab[P - p0, T - t0] - 1
    cid = np.full(n, -1, np.int64)
    idmask = np.zeros(m, np.int64); nhit = np.zeros(m, np.int64)
    dt = np.full(m, np.nan); dp = np.full(m, np.nan)
    if m == 0:
        return cid, comp, idmask, nhit, dt, dp

    pc, tc = c["pad_c"] + poff, c["tbin"] + toff
    ps, zs = c["phisize"], c["zsize"]
    bp0 = np.rint(pc - (ps - 1) / 2).astype(int); bp1 = bp0 + ps - 1
    bt0 = np.rint(tc - (zs - 1) / 2).astype(int); bt1 = bt0 + zs - 1
    Pc, Tc = P[:, None], T[:, None]
    exact = (Pc >= bp0) & (Pc <= bp1) & (Tc >= bt0) & (Tc <= bt1)
    wide = (Pc >= bp0 - 1) & (Pc <= bp1 + 1) & (Tc >= bt0 - 1) & (Tc <= bt1 + 1)
    d2 = ((Pc - pc) / np.maximum(ps, 1)) ** 2 + ((Tc - tc) / np.maximum(zs, 1)) ** 2
    score = d2 + np.where(exact, 0.0, 10.0)
    deficit = c["adc"].astype(np.float64).copy()
    maxadc = c["maxadc"]

    for j in np.argsort(-c["adc"]):
        if anchor_rel_tol is None:
            anchor_ok = (A == maxadc[j])
        else:
            anchor_ok = np.abs(A - maxadc[j]) <= anchor_rel_tol * np.abs(maxadc[j])
        cand = np.where(wide[:, j] & anchor_ok & (cid < 0))[0]
        if len(cand):
            i = cand[np.argmin(score[cand, j])]
            cid[i] = j; deficit[j] -= A[i]
    for i in np.argsort(-A):
        if cid[i] >= 0:
            continue
        ok = wide[i] & (deficit >= A[i] * (1 - REL_TOL))
        if ok.any():
            j = int(np.argmin(np.where(ok, score[i], np.inf)))
            cid[i] = j; deficit[j] -= A[i]
    for j in range(m):
        sel = np.where(cid == j)[0]
        nhit[j] = len(sel)
        if not len(sel):
            continue
        ph, tb, ad = P[sel], T[sel], A[sel]
        w = ad.sum()
        dt[j] = (ad * tb).sum() / w - tc[j]
        dp[j] = (ad * ph).sum() / w - pc[j]
        cl_adc = c["adc"][j]
        b = 1 * (abs(w - cl_adc) <= REL_TOL * abs(cl_adc))
        b |= 2 * (abs(ad.max() - maxadc[j]) <= REL_TOL * abs(maxadc[j]))
        b |= 4 * ((ph.max() - ph.min() + 1 == ps[j]) and (tb.max() - tb.min() + 1 == zs[j]))
        b |= 8 * ((abs(dt[j]) < CENTROID_TOL_T) and (abs(dp[j]) < CENTROID_TOL_PAD))
        idmask[j] = b
    return cid, comp, idmask, nhit, dt, dp


def label_event_sim(h, c, dead, toff, poff, anchor_rel_tol):
    keys = ["layer", "phielem", "zelem"]
    hg, cg = h.groupby(keys).indices, c.groupby(keys).indices
    cid = np.full(len(h), -1, np.int64); comp = np.zeros(len(h), np.int64); hs = np.zeros(len(h), np.int64)
    idmask = np.zeros(len(c), np.int64); nhit = np.zeros(len(c), np.int64)
    dt = np.full(len(c), np.nan); dp = np.full(len(c), np.nan)
    for k_i, (k, hi) in enumerate(hg.items()):
        ci = cg.get(k, np.array([], np.int64))
        ci = ci[np.isfinite(c.pad_c.values[ci]) & c.finite.values[ci]]
        cd = {col: c[col].values[ci] for col in ["pad_c", "tbin", "adc", "maxadc", "phisize", "zsize"]}
        lcid, lcomp, lidmask, lnhit, ldt, ldp = label_hitset_sim(
            h.phibin.values[hi], h.tbin.values[hi], h.adc.values[hi], cd, toff, poff, anchor_rel_tol)
        if len(ci):
            cid[hi] = np.where(lcid >= 0, c.cl_id.values[ci][np.maximum(lcid, 0)], -1)
        else:
            cid[hi] = -1
        comp[hi] = lcomp; hs[hi] = k_i
        idmask[ci] = lidmask; nhit[ci] = lnhit; dt[ci] = ldt; dp[ci] = ldp
    h["cid"], h["comp"], h["hs"] = cid, comp, hs
    c["idmask"], c["nhit"], c["dt"], c["dp"] = idmask, nhit, dt, dp
    c["tbin_sentinel"] = (c.tbin.values < -0.5).astype(np.uint8)
    c["recovered"] = (((idmask & 7) == 7) & (((idmask & 8) > 0) | (c.tbin_sentinel.values > 0))).astype(np.uint8)

    h["comp_n"] = h.groupby(["hs", "comp"])["adc"].transform("size").astype(np.int64)
    pairs = h.loc[h.cid >= 0, ["hs", "comp", "cid"]].drop_duplicates()
    ncl = pairs.groupby(["hs", "comp"]).size().rename("comp_ncl")
    h = h.join(ncl, on=["hs", "comp"]); h["comp_ncl"] = h.comp_ncl.fillna(0).astype(np.int64)
    ccomp = pairs.groupby("cid").agg(comp=("comp", "first"), hs=("hs", "first"), ncomp=("comp", "nunique"))
    c = c.join(ccomp, on="cl_id")
    c["comp"] = c.comp.fillna(-1).astype(np.int64)
    c["hs"] = c.hs.fillna(-1).astype(np.int64)
    c["ncomp"] = c.ncomp.fillna(0).astype(np.int64)
    c = c.join(ncl, on=["hs", "comp"]); c["comp_ncl"] = c.comp_ncl.fillna(0).astype(np.int64)

    reg = region(h.layer.values)
    ZS_THR = {1: 11, 2: 21, 3: 21}
    h["in_drift_window"] = (h.tbin >= 0) & (h.tbin <= 251)
    h["subthr"] = h.adc.values < np.array([ZS_THR[r] for r in (1, 2, 3)])[reg - 1]
    nps = np.array([NPAD[L] for L in h.layer.values]) // NSEC
    lp = h.phibin.values - h.phielem.values * nps
    h["edgepad"] = (lp == 0) | (lp == nps - 1)
    da = np.zeros(len(h), bool)
    for k, hi in hg.items():
        d = dead.get(k)
        if d is None:
            continue
        l = lp[hi]
        da[hi] = (d[np.clip(l - 1, 0, len(d) - 1)] & (l > 0)) | (d[np.clip(l + 1, 0, len(d) - 1)] & (l < len(d) - 1))
    h["deadadj"] = da

    h = h.merge(c[["cl_id", "idmask", "recovered"]], left_on="cid", right_on="cl_id", how="left").drop(columns="cl_id")
    h["idmask"] = h.idmask.fillna(0).astype(np.int64)
    h["recovered"] = h.recovered.fillna(0).astype(np.uint8)
    return h, c


def run_once(hfile, table, dead, cfile, event, toff, poff, anchor_rel_tol):
    h = read_sim_hits(hfile, event)
    c = load_sim_clusters(cfile, event)
    c["ontrack"] = False
    c = add_pad_centroid(c, table)
    h, c = label_event_sim(h, c, dead, toff, poff, anchor_rel_tol)
    return h, c


# ----------------------------------------------------------------------------- geometry-only association
def geometric_islands(h, c, toff, poff):
    """Pure geometry: per-hitset cc_label islands on hits; each cluster -> the island of
    the nearest hit (by the same normalised-distance score, no adc matching) inside its
    widened box, or -1 if no hit falls in the widened box."""
    keys = ["layer", "phielem", "zelem"]
    hg, cg = h.groupby(keys).indices, c.groupby(keys).indices
    comp = np.zeros(len(h), np.int64); hs = np.zeros(len(h), np.int64)
    assoc_comp = np.full(len(c), -1, np.int64); assoc_hs = np.full(len(c), -1, np.int64)
    for k_i, (k, hi) in enumerate(hg.items()):
        P, T = h.phibin.values[hi], h.tbin.values[hi]
        p0, t0 = P.min(), T.min()
        img = np.zeros((P.max() - p0 + 1, T.max() - t0 + 1), np.int8)
        img[P - p0, T - t0] = 1
        lab, _ = cc_label(img)
        lcomp = lab[P - p0, T - t0] - 1
        comp[hi] = lcomp; hs[hi] = k_i

        ci = cg.get(k, np.array([], np.int64))
        ci = ci[np.isfinite(c.pad_c.values[ci]) & c.finite.values[ci]]
        if not len(ci):
            continue
        pc, tc = c.pad_c.values[ci] + poff, c.tbin.values[ci] + toff
        ps, zs = c.phisize.values[ci], c.zsize.values[ci]
        bp0 = np.rint(pc - (ps - 1) / 2).astype(int); bp1 = bp0 + ps - 1
        bt0 = np.rint(tc - (zs - 1) / 2).astype(int); bt1 = bt0 + zs - 1
        Pc, Tc = P[:, None], T[:, None]
        wide = (Pc >= bp0 - 1) & (Pc <= bp1 + 1) & (Tc >= bt0 - 1) & (Tc <= bt1 + 1)
        d2 = ((Pc - pc) / np.maximum(ps, 1)) ** 2 + ((Tc - tc) / np.maximum(zs, 1)) ** 2
        for jj, cl_idx in enumerate(ci):
            cand = np.where(wide[:, jj])[0]
            if not len(cand):
                continue
            i = cand[np.argmin(d2[cand, jj])]
            assoc_comp[cl_idx] = lcomp[i]; assoc_hs[cl_idx] = k_i
    h = h.copy(); c = c.copy()
    h["comp_geo"], h["hs_geo"] = comp, hs
    c["assoc_comp"], c["assoc_hs"] = assoc_comp, assoc_hs
    valid = c.assoc_comp.values >= 0
    tmp = pd.DataFrame({"hs": c.assoc_hs.values[valid], "comp": c.assoc_comp.values[valid]})
    cnt = tmp.groupby(["hs", "comp"]).size().reset_index(name="assoc_ncl")
    c = c.merge(cnt, left_on=["assoc_hs", "assoc_comp"], right_on=["hs", "comp"], how="left").drop(columns=["hs", "comp"])
    c["assoc_ncl"] = c.assoc_ncl.fillna(0).astype(np.int64)
    return h, c


def single_cluster_island_stats(h, c, h_hs_col, h_comp_col, c_hs_col, c_comp_col, ncl_col,
                                 adc_col="adc", maxadc_col="maxadc"):
    """h carries per-hit island ids under (h_hs_col, h_comp_col); c carries the per-cluster
    association under (c_hs_col, c_comp_col) (may be differently named, e.g. sim's assoc_hs/
    assoc_comp vs the real pipeline's own hs/comp) plus the per-island cluster count (ncl_col)."""
    isl = h.groupby([h_hs_col, h_comp_col]).agg(
        isl_adc_sum=("adc", "sum"), isl_adc_max=("adc", "max"),
        isl_p_min=("phibin", "min"), isl_p_max=("phibin", "max"),
        isl_t_min=("tbin", "min"), isl_t_max=("tbin", "max"),
    ).reset_index()
    single = c[c[ncl_col] == 1].merge(isl, left_on=[c_hs_col, c_comp_col], right_on=[h_hs_col, h_comp_col], how="left")
    out = pd.DataFrame({
        "R_sum": single["isl_adc_sum"] / single[adc_col],
        "R_max": single["isl_adc_max"] / single[maxadc_col],
        "D_max": single["isl_adc_max"] - single[maxadc_col],
        "dphisize": (single["isl_p_max"] - single["isl_p_min"] + 1) - single["phisize"],
        "dzsize": (single["isl_t_max"] - single["isl_t_min"] + 1) - single["zsize"],
    })
    return out, len(single)


def quantile_report(df):
    return {col: {f"q{int(q*100):02d}": float(df[col].quantile(q)) for q in QUANTILES} for col in df.columns}


# ----------------------------------------------------------------------------- helpers for the side-by-side table
def island_shape_stats(h, hs_col, comp_col):
    """share of 4-connected islands that are single-hit, for whichever comp/hs columns are given."""
    sizes = h.groupby([hs_col, comp_col]).size()
    return float(np.mean(sizes.values == 1)), int(len(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{LABELS_DIR}/sim_v61")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ledger = f"{LABELS_DIR}/RESULTS_LEDGER.txt"

    hpath = os.path.join(TPC_SIM_DIR, "hit69_frames_production_v61.root")
    cpath = os.path.join(TPC_SIM_DIR, "island91_frames_production_v61.root")
    hfile, cfile = uproot.open(hpath), uproot.open(cpath)
    EVENT = 0

    table = build_pad_table(hfile); dead = dead_map(table)
    print(f"pad table: {len(table)} hitsets")

    lines = [f"\n=== Phase 3 (re-scoped): sim event {EVENT} ==="]

    # ---------------- integer-ness of the sim adc values -----------------------------------------------------
    h_raw = read_sim_hits(hfile, EVENT)
    c_raw_all = cfile["ntp_cluster"].arrays(["event", "adc", "maxadc"], library="pd")
    c_raw = c_raw_all[c_raw_all.event == EVENT]
    def intness(v):
        v = v[np.isfinite(v)]
        frac0 = float(np.mean(np.isclose(v, np.round(v))))
        return frac0, float(v.min()), float(v.max())
    hit_i = intness(h_raw.adc.values)
    cadc_i = intness(c_raw.adc.values)
    cmax_i = intness(c_raw.maxadc.values)
    lines.append(f"3c-1 fraction zero-fractional-part (min,max): hit adc {hit_i[0]:.4f} ({hit_i[1]:.3f},{hit_i[2]:.3f}); "
                 f"cluster adc {cadc_i[0]:.4f} ({cadc_i[1]:.3f},{cadc_i[2]:.3f}); "
                 f"cluster maxadc {cmax_i[0]:.4f} ({cmax_i[1]:.3f},{cmax_i[2]:.3f})")
    print(lines[-1])

    # ---------------- calibration run (feeds the toff/poff of the geometric association; anchor tolerance = REL_TOL) ----
    h1, c1 = run_once(hfile, table, dead, cfile, EVENT, 0.0, 0.0, REL_TOL)
    ok = (c1.idmask.values & 1) > 0
    n_ok = int(ok.sum())
    dt_med = float(np.nanmedian(c1.dt.values[ok])) if n_ok else float("nan")
    dp_med = float(np.nanmedian(c1.dp.values[ok])) if n_ok else float("nan")
    stable = n_ok >= MIN_STABLE_N and np.isfinite(dt_med) and np.isfinite(dp_med)
    if stable and (abs(dt_med) > 0.2 or abs(dp_med) > 0.2):
        toff, poff = dt_med, dp_med
        rerun_note = f"rerun with toff={toff:+.4f} poff={poff:+.4f} (n bit1-passing={n_ok})"
    elif stable:
        toff, poff = 0.0, 0.0
        rerun_note = f"no rerun needed: |median dt|={dt_med:+.4f} |median dp|={dp_med:+.4f} both <=0.2 bin (n bit1-passing={n_ok})"
    else:
        toff, poff = 0.0, 0.0
        rerun_note = (f"calibration UNSTABLE: only {n_ok} bit1-passing clusters (< MIN_STABLE_N={MIN_STABLE_N}), "
                       f"median dt={dt_med!r} dp={dp_med!r} -- using toff=poff=0")
    lines.append(f"3c-2 calibration: {rerun_note}")
    print(lines[-1])

    # ---------------- pure-geometry association on sim ----------------------------
    h_geo, c_geo = geometric_islands(h1[["layer", "phielem", "zelem", "phibin", "tbin", "adc"]].copy(),
                                      c1[["layer", "phielem", "zelem", "pad_c", "tbin", "finite",
                                          "phisize", "zsize", "adc", "maxadc", "cls"]].copy(),
                                      toff, poff)
    frac_no_hit_in_box = float(np.mean(c_geo.assoc_comp.values < 0))
    lines.append(f"3c-2 share of clusters with no hit in widened box: {frac_no_hit_in_box:.4f} (n_clusters={len(c_geo)})")
    print(lines[-1])

    sim_single, n_sim_single = single_cluster_island_stats(h_geo, c_geo, "hs_geo", "comp_geo", "assoc_hs", "assoc_comp", "assoc_ncl")
    lines.append(f"3c-3 SIM single-cluster islands (n={n_sim_single}, of {int((c_geo.assoc_ncl.values==1).sum())} clusters):")
    lines.append(f"  {quantile_report(sim_single)}")
    print(lines[-2]); print(lines[-1])

    # ---------------- REAL event 1 reference (existing recovered-membership comp/hs/comp_ncl) ----
    h_r1 = uproot.open(f"{REAL_OUT}/hits_ev001.root")["hits"].arrays(
        ["hs", "comp", "adc", "phibin", "tbin"], library="pd")
    c_r1 = uproot.open(f"{REAL_OUT}/clusters_ev001.root")["clusters"].arrays(
        ["hs", "comp", "comp_ncl", "adc", "maxadc", "phisize", "zsize"], library="pd")
    real_single, n_real_single = single_cluster_island_stats(h_r1, c_r1, "hs", "comp", "hs", "comp", "comp_ncl")
    lines.append(f"3c-3 REAL event 1 reference (n={n_real_single}, of {int((c_r1.comp_ncl.values==1).sum())} clusters):")
    lines.append(f"  {quantile_report(real_single)}")
    print(lines[-2]); print(lines[-1])

    # ---------------- baseline recovery run (1e-3 on bits 1,2, budget AND anchor) ------
    h_base, c_base = h1, c1   # already run with anchor_rel_tol=REL_TOL and calibrated toff/poff=0/0 pass 1;
    if (toff, poff) != (0.0, 0.0):   # if calibration called for a rerun, use that run for the baseline too
        h_base, c_base = run_once(hfile, table, dead, cfile, EVENT, toff, poff, REL_TOL)
    frac_recovered = float(np.mean(c_base.recovered.values == 1))
    frac_assigned = float(np.mean(h_base.cid.values >= 0))
    idmask_hist = {int(k): int(v) for k, v in zip(*np.unique(c_base.idmask.values, return_counts=True))}
    by_cls = c_base.groupby("cls")["recovered"].mean().to_dict()
    lines.append(f"3c-4 baseline, not tuned (toff={toff:+.4f}, poff={poff:+.4f}, REL_TOL={REL_TOL}, anchor REL_TOL={REL_TOL}):")
    lines.append(f"  n_hits={len(h_base)} n_clusters={len(c_base)} frac_recovered={frac_recovered:.4f} frac_assigned={frac_assigned:.4f}")
    lines.append(f"  idmask histogram: {dict(sorted(idmask_hist.items(), key=lambda kv: -kv[1]))}")
    lines.append(f"  frac_recovered per truth class (cls 0=track 1=looper 2=noise): {by_cls}")
    for l in lines[-4:]:
        print(l)

    write_root_tree_fixed(f"{a.out}/hits_ev{EVENT:03d}.root", "hits", h_base, HITS_SCHEMA_SIM)
    write_root_tree_fixed(f"{a.out}/clusters_ev{EVENT:03d}.root", "clusters", c_base, CLUSTERS_SCHEMA_SIM)

    # ---------------- side-by-side table --------------------------------------------------
    real25_h = uproot.open(f"{REAL_OUT}/hits_ev025.root")["hits"].arrays(
        ["hs", "comp", "adc", "phibin", "tbin"], library="pd")
    real25_c = uproot.open(f"{REAL_OUT}/clusters_ev025.root")["clusters"].arrays(
        ["hs", "comp", "comp_ncl", "phisize", "zsize"], library="pd")

    single_hit_share_sim, n_isl_sim = island_shape_stats(h_geo, "hs_geo", "comp_geo")
    single_hit_share_r1, n_isl_r1 = island_shape_stats(h_r1, "hs", "comp")
    single_hit_share_r25, n_isl_r25 = island_shape_stats(real25_h, "hs", "comp")

    def qtab(vals, qs=(0.50, 0.90, 0.99)):
        return {f"q{int(q*100)}": float(np.quantile(vals, q)) for q in qs}

    rec_sim = dict(
        label="sim ev0", n_hits=len(h_base), n_clusters=len(c_base),
        hits_per_cluster_mean=len(h_base) / len(c_base) if len(c_base) else float("nan"),
        hits_per_cluster_median=float(c_base.nhit.median()) if "nhit" in c_base else float("nan"),
        share_single_hit_islands=single_hit_share_sim,
        share_islands_ge2cl=float(np.mean(c_geo.assoc_ncl.values >= 2)),
        phisize_q=qtab(c_base.phisize.values), zsize_q=qtab(c_base.zsize.values),
        frac_recovered_per_cls=by_cls,
        truth_alignment="ntp_truth confirmed entry-aligned with ntp_cluster (3b: identical event arrays, equal entry counts)",
    )
    rec_r1 = dict(
        label="real ev1", n_hits=len(h_r1), n_clusters=len(c_r1),
        hits_per_cluster_mean=len(h_r1) / len(c_r1) if len(c_r1) else float("nan"),
        # hits_per_cluster_median filled below from clusters_ev001.root's own nhit column
        share_single_hit_islands=single_hit_share_r1,
        share_islands_ge2cl=float(np.mean(c_r1.comp_ncl.values >= 2)),
        phisize_q=qtab(c_r1.phisize.values), zsize_q=qtab(c_r1.zsize.values),
    )
    rec_r25 = dict(
        label="real ev25", n_hits=len(real25_h), n_clusters=len(real25_c),
        hits_per_cluster_mean=len(real25_h) / len(real25_c) if len(real25_c) else float("nan"),
        share_single_hit_islands=single_hit_share_r25,
        share_islands_ge2cl=float(np.mean(real25_c.comp_ncl.values >= 2)),
        phisize_q=qtab(real25_c.phisize.values), zsize_q=qtab(real25_c.zsize.values),
    )
    # hits-per-cluster median for the real entries: use nhit from clusters_evNNN.root directly (already stored)
    c_r1_nhit = uproot.open(f"{REAL_OUT}/clusters_ev001.root")["clusters"].arrays(["nhit"], library="pd")
    c_r25_nhit = uproot.open(f"{REAL_OUT}/clusters_ev025.root")["clusters"].arrays(["nhit"], library="pd")
    rec_r1["hits_per_cluster_median"] = float(c_r1_nhit.nhit.median())
    rec_r25["hits_per_cluster_median"] = float(c_r25_nhit.nhit.median())

    lines.append("3d side-by-side table:")
    for rec in (rec_sim, rec_r1, rec_r25):
        lines.append(f"  {rec}")
    for l in lines[-4:]:
        print(l)

    with open(ledger, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nledger updated -> {ledger}")


if __name__ == "__main__":
    main()
