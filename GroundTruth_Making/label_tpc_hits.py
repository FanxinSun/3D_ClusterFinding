#!/usr/bin/env python
"""
label_tpc_hits.py  ntuplizer.root  [--events 1-3] [--out data/labels_v0] [--toff 0] [--poff 0]

Per-hit labels for sPHENIX TPC hits, reconstructed from the ntuplizer's own
ntp_cluster / ntp_clus_trk trees and certified with the integer cluster summaries.
Writes  <out>/hits_evNNN.root (tree hits), <out>/clusters_evNNN.root (tree clusters),
        <out>/offsets_evNNN.csv, <out>/event_summary.csv, <out>/review_hitsets.csv

Adapted from an earlier prototype. What differs from it:
  1. label_event: the crash when a hitset has hits but no clusters (ci empty)
     is guarded -- indexing c.cl_id.values[ci][...] on an empty array raised
     IndexError even though np.where would have discarded the result. Such
     hits now get cid = -1 directly.
  2. Output is per-event ROOT via root_io_utils instead of parquet; hits gain
     an `ientry` column (int32 entry index into the original ntp_hit) and
     drop `phi`.
  3. Column and function names differ: idmask (prototype: cert),
     in_drift_window (intime), review_table (review_rows), n_notfull_cl
     (n_uncert_cl). The prototype's per-hit rank column (assigned +
     fully-identified + on-track + clean) is not persisted (derivable from
     cid, idmask, ontrack, clean); the fraction of hits reaching that rank is
     still reported as a summary statistic (frac_clean_ontrack_full),
     computed locally.
  4. Event 44 (GL1 laser event) is processed like any other event but flagged
     laser_event=1 in event_summary.csv and skipped in every cross-event
     aggregate.
  5. event_summary.csv holds one entry per event.
  6. Extra diagnostics: per-(layer,sector,side) dt/dp offset table
     (offsets_evNNN.csv).

Identification and selection:
  - Two centroid tolerances: CENTROID_TOL_T=0.10 bin, CENTROID_TOL_PAD=0.02
    pad. Bit 8 of idmask = both satisfied.
  - Cluster columns tbin_sentinel (stored tbin < -0.5) and recovered (bits
    1,2,4 all set AND (bit 8 set OR tbin_sentinel)); recovered is propagated
    to hits. idmask is kept as-is; recovered (not idmask==15) is the SELECTION
    used everywhere downstream (descriptors, plots, aggregates, review
    scoring, gate).
  - --toff/--poff are fixed for the whole run (no per-event recalibration);
    the per-event median dt of bits-1,2,4-passing clusters with a valid
    (non-sentinel) tbin is reported as dt_median_resid.
  - Fixed output dtype schema (HITS_SCHEMA / CLUSTERS_SCHEMA in root_io_utils)
    instead of a per-event automatic downcast, so per-event files
    concatenate; root_io_utils.write_root_tree_fixed stops and raises instead
    of silently widening a column that does not fit.
  - event_summary.csv carries tbin_q50, tbin_q99, frac_recovered,
    n_fail_single/n_single, n_fail_multi/n_multi, frac_fail_streak,
    dt_median_resid.
  - --gate-check (event 1 only) passes if frac_recovered >= 0.85.

hits.adc is int32 (not int16, not uint16 -- unsigned columns wrap silently in
later arithmetic). Hits keep an `adc_anomalous` (uint8) flag = adc > 1023 (a
10-bit ADC cannot produce a larger value); such hits are kept, not excluded or
altered; cluster_descriptors.py skips any cluster containing one.
"""
import argparse, os, time
import numpy as np, pandas as pd, uproot
from scipy.ndimage import label as cc_label

from root_io_utils import write_root_tree, write_root_tree_fixed, HITS_SCHEMA, CLUSTERS_SCHEMA

LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels_v0")   # product directory; override on a new machine

# ----------------------------------------------------------------------------- constants from the probes
NPAD   = {L: (1128 if L < 23 else 1536 if L < 39 else 2304) for L in range(7, 55)}  # R1 = 94 pads / sector
NSEC   = 12
ZS_THR = {1: 11, 2: 21, 3: 21}          # zero-suppression floor per region
ADC_SAT = 963
TB_HALF_DRIFT = 251                      # 106.28 cm / 0.4234 cm per tbin
CLEAN  = dict(sntpc=30, ptmin=0.15, ptmax=10., etamax=1.1, reszmax=1.0)
CENTROID_TOL_T = 0.10                    # bins, time-centroid check
CENTROID_TOL_PAD = 0.02                  # pad, pad-centroid check
FIXED_TOFF = 0.6961883752                # fixed for the whole run, no per-event recalibration
FIXED_POFF = 0.0
ADC_10BIT_MAX = 1023                     # a 10-bit ADC cannot produce a larger value
STREAK_ZSIZE, STREAK_PHISIZE = 15, 8     # frac_fail_streak thresholds
LASER_EVENT = 44                         # GL1 laser event: processed, excluded from aggregates
HIT_COLS = ["layer", "phielem", "zelem", "phibin", "tbin", "adc"]   # phi dropped from output
CL_COLS  = ["event", "layer", "phielem", "zelem", "phi", "tbin", "adc", "maxadc",
            "phisize", "zsize", "pedge", "x", "y", "z"]
TRK_COLS = ["event", "seedID", "x", "y", "z", "sntpc", "spt", "seta", "resz", "resphi",
            "alpha", "beta", "scharge", "sdedx"]
DUP_COLS = ["event", "layer", "phielem", "zelem", "phi", "tbin", "adc", "maxadc",
            "phisize", "zsize", "pedge", "x", "y", "z"]

def region(layer):
    return np.where(layer < 23, 1, np.where(layer < 39, 2, 3))

# ----------------------------------------------------------------------------- clusters + track labels
def _xyz_keys(d, ok):
    out = []
    for a in ("x", "y", "z"):
        v = d[a].values.astype(np.float64)
        k = np.full(len(v), np.iinfo(np.int64).min, np.int64)
        k[ok] = np.rint(v[ok] * 1e4).astype(np.int64)          # 1 um
        out.append(k)
    return out

def load_clusters(f):
    c = f["ntp_cluster"].arrays(CL_COLS, library="pd")
    c = c[(c.layer >= 7) & (c.layer <= 54)].reset_index(drop=True)
    c["cl_id"] = np.arange(len(c), dtype=np.int64)
    c["finite"] = np.isfinite(c.x) & np.isfinite(c.y) & np.isfinite(c.z) & np.isfinite(c.tbin) & np.isfinite(c.phi)
    for k in ["event", "layer", "phielem", "zelem", "adc", "maxadc", "phisize", "zsize", "pedge"]:
        c[k] = c[k].astype(np.int64)
    c["ix"], c["iy"], c["iz"] = _xyz_keys(c, c.finite.values)

    t = f["ntp_clus_trk"].arrays(TRK_COLS, library="pd")
    t["clean"] = ((t.sntpc >= CLEAN["sntpc"]) & (t.spt > CLEAN["ptmin"]) & (t.spt < CLEAN["ptmax"])
                  & (t.seta.abs() < CLEAN["etamax"]) & (t.resz.abs() < CLEAN["reszmax"]))
    t["ix"], t["iy"], t["iz"] = _xyz_keys(t, np.ones(len(t), bool))
    tg = (t.groupby(["event", "ix", "iy", "iz"])
            .agg(ntrk_share=("seedID", "size"), seedID=("seedID", "first"), clean=("clean", "max"),
                 spt=("spt", "first"), seta=("seta", "first"), alpha=("alpha", "first"),
                 beta=("beta", "first"), resz=("resz", "first"), resphi=("resphi", "first"))
            .reset_index())
    c = c.merge(tg, on=["event", "ix", "iy", "iz"], how="left")
    c["ontrack"] = c.ntrk_share.notna()
    c["ntrk_share"] = c.ntrk_share.fillna(0).astype(np.int64)
    c["clean"] = c.clean.fillna(False).astype(bool)
    c["seedID"] = c.seedID.fillna(-1).astype(np.int64)
    return c

# ----------------------------------------------------------------------------- pad geometry from the hits
def build_pad_table(f, nrows=5_000_000):
    """(layer, sector, side) -> (pads, unwrapped phi, ref).  Built from the hits, so it is exactly
    the geometry the ntuplizer used.  Pads absent here are dead (>=20 hits/pad expected)."""
    d = f["ntp_hit"].arrays(["layer", "phielem", "zelem", "phibin", "phi"], entry_stop=nrows, library="pd")
    d = d[(d.layer >= 7) & (d.layer <= 54) & np.isfinite(d.phibin) & np.isfinite(d.phi)]
    d = d.astype({"layer": int, "phielem": int, "zelem": int, "phibin": int})
    d = d.drop_duplicates(["layer", "phielem", "zelem", "phibin"])
    table = {}
    for k, g in d.groupby(["layer", "phielem", "zelem"]):
        ref = np.angle(np.exp(1j * g.phi.values).mean())
        u = ref + np.angle(np.exp(1j * (g.phi.values - ref)))     # unwrap around the sector
        o = np.argsort(u)
        table[k] = (g.phibin.values[o].astype(np.float64), u[o], ref)
    return table

def dead_map(table):
    dead = {}
    for (L, s, side), (pads, u, ref) in table.items():
        n = NPAD[L] // NSEC
        present = np.zeros(n, bool)
        lp = pads.astype(int) - s * n
        present[lp[(lp >= 0) & (lp < n)]] = True
        dead[(L, s, side)] = ~present
    return dead

def add_pad_centroid(c, table):
    pad_c = np.full(len(c), np.nan)
    for k, idx in c.groupby(["layer", "phielem", "zelem"]).indices.items():
        if k not in table:
            continue
        pads, u, ref = table[k]
        uu = ref + np.angle(np.exp(1j * (c.phi.values[idx] - ref)))
        pad_c[idx] = np.interp(uu, u, pads)
    c["pad_c"] = pad_c
    return c

# ----------------------------------------------------------------------------- hits
def event_bounds(f):
    ev = f["ntp_hit"]["event"].array(library="np").astype(np.int64)     # tree is sorted by event (probe_ntuplizer.py, section [2i])
    return np.searchsorted(ev, np.arange(1, ev.max() + 2)), int(ev.max())

def read_event_hits(f, starts, e):
    lo, hi = int(starts[e - 1]), int(starts[e])
    h = f["ntp_hit"].arrays(HIT_COLS, entry_start=lo, entry_stop=hi, library="pd")
    h["ientry"] = np.arange(lo, hi, dtype=np.int64)
    m = (h.layer >= 7) & (h.layer <= 54) & np.isfinite(h.phibin) & np.isfinite(h.tbin)
    h = h[m].copy()
    for k in ["layer", "phielem", "zelem", "phibin", "tbin", "adc"]:
        h[k] = h[k].astype(np.int64)
    h.insert(0, "event", e)
    return h.reset_index(drop=True)

# ----------------------------------------------------------------------------- core: one hitset
def label_hitset(P, T, A, c, toff=0.0, poff=0.0):
    """P, T, A: int arrays (phibin, tbin, adc) of the hits of one hitset.
    c: dict of arrays for the clusters of the same hitset (pad_c, tbin, adc, maxadc, phisize, zsize).
    Returns cid (local cluster index or -1), comp, and per-cluster idmask / nhit / dt / dp."""
    n, m = len(P), len(c["adc"])
    p0, t0 = P.min(), T.min()
    img = np.zeros((P.max() - p0 + 1, T.max() - t0 + 1), np.int8)
    img[P - p0, T - t0] = 1
    lab, _ = cc_label(img)                                   # 4-connectivity
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
    wide  = (Pc >= bp0 - 1) & (Pc <= bp1 + 1) & (Tc >= bt0 - 1) & (Tc <= bt1 + 1)
    d2 = ((Pc - pc) / np.maximum(ps, 1)) ** 2 + ((Tc - tc) / np.maximum(zs, 1)) ** 2
    score = d2 + np.where(exact, 0.0, 10.0)                   # exact box strongly preferred
    deficit = c["adc"].astype(np.int64).copy()

    # 1) anchors: the maxadc hit of every cluster
    for j in np.argsort(-c["adc"]):
        cand = np.where(wide[:, j] & (A == c["maxadc"][j]) & (cid < 0))[0]
        if len(cand):
            i = cand[np.argmin(score[cand, j])]
            cid[i] = j; deficit[j] -= A[i]
    # 2) greedy fill under the ADC budget, largest hits first
    for i in np.argsort(-A):
        if cid[i] >= 0:
            continue
        ok = wide[i] & (deficit >= A[i])
        if ok.any():
            j = int(np.argmin(np.where(ok, score[i], np.inf)))
            cid[i] = j; deficit[j] -= A[i]
    # 3) certification
    for j in range(m):
        sel = np.where(cid == j)[0]
        nhit[j] = len(sel)
        if not len(sel):
            continue
        ph, tb, ad = P[sel], T[sel], A[sel]
        w = ad.sum()
        dt[j] = (ad * tb).sum() / w - tc[j]
        dp[j] = (ad * ph).sum() / w - pc[j]
        b  = 1 * (w == c["adc"][j])
        b |= 2 * (ad.max() == c["maxadc"][j])
        b |= 4 * ((ph.max() - ph.min() + 1 == ps[j]) and (tb.max() - tb.min() + 1 == zs[j]))
        b |= 8 * ((abs(dt[j]) < CENTROID_TOL_T) and (abs(dp[j]) < CENTROID_TOL_PAD))
        idmask[j] = b
    return cid, comp, idmask, nhit, dt, dp

# ----------------------------------------------------------------------------- one event
def label_event(h, c, dead, toff=0.0, poff=0.0):
    keys = ["layer", "phielem", "zelem"]
    hg, cg = h.groupby(keys).indices, c.groupby(keys).indices
    cid = np.full(len(h), -1, np.int64); comp = np.zeros(len(h), np.int64); hs = np.zeros(len(h), np.int64)
    idmask = np.zeros(len(c), np.int64); nhit = np.zeros(len(c), np.int64)
    dt = np.full(len(c), np.nan); dp = np.full(len(c), np.nan)
    for k_i, (k, hi) in enumerate(hg.items()):
        ci = cg.get(k, np.array([], np.int64))
        ci = ci[np.isfinite(c.pad_c.values[ci]) & c.finite.values[ci]]
        cd = {col: c[col].values[ci] for col in ["pad_c", "tbin", "adc", "maxadc", "phisize", "zsize"]}
        lcid, lcomp, lidmask, lnhit, ldt, ldp = label_hitset(
            h.phibin.values[hi], h.tbin.values[hi], h.adc.values[hi], cd, toff, poff)
        if len(ci):
            cid[hi] = np.where(lcid >= 0, c.cl_id.values[ci][np.maximum(lcid, 0)], -1)
        else:
            cid[hi] = -1   # hitset has hits but no (finite) clusters -- nothing to assign to
        comp[hi] = lcomp; hs[hi] = k_i
        idmask[ci] = lidmask; nhit[ci] = lnhit; dt[ci] = ldt; dp[ci] = ldp
    h["cid"], h["comp"], h["hs"] = cid, comp, hs
    c["idmask"], c["nhit"], c["dt"], c["dp"] = idmask, nhit, dt, dp
    c["tbin_sentinel"] = (c.tbin.values < -0.5).astype(np.uint8)                          # tbin below -0.5 is a sentinel
    c["recovered"] = (((idmask & 7) == 7) & (((idmask & 8) > 0) | (c.tbin_sentinel.values > 0))).astype(np.uint8)  # bits 1,2,4 set, and bit 8 set or a sentinel tbin

    # component bookkeeping
    h["comp_n"] = h.groupby(["hs", "comp"])["adc"].transform("size").astype(np.int64)
    pairs = h.loc[h.cid >= 0, ["hs", "comp", "cid"]].drop_duplicates()
    ncl = pairs.groupby(["hs", "comp"]).size().rename("comp_ncl")
    h = h.join(ncl, on=["hs", "comp"]); h["comp_ncl"] = h.comp_ncl.fillna(0).astype(np.int64)
    ccomp = pairs.groupby("cid").agg(comp=("comp", "first"), hs=("hs", "first"), ncomp=("comp", "nunique"))
    c = c.join(ccomp, on="cl_id")
    # clusters with nhit==0 (no assigned hits) are absent from `pairs` and so get NaN comp/hs/ncomp
    # from the join above; fill with the same not-in-any-island sentinel used elsewhere (-1 / 0).
    c["comp"] = c.comp.fillna(-1).astype(np.int64)
    c["hs"] = c.hs.fillna(-1).astype(np.int64)
    c["ncomp"] = c.ncomp.fillna(0).astype(np.int64)
    c = c.join(ncl, on=["hs", "comp"]); c["comp_ncl"] = c.comp_ncl.fillna(0).astype(np.int64)

    # flags
    reg = region(h.layer.values)
    h["in_drift_window"] = (h.tbin >= 0) & (h.tbin <= TB_HALF_DRIFT)
    h["subthr"] = h.adc.values < np.array([ZS_THR[r] for r in (1, 2, 3)])[reg - 1]
    h["satur"] = h.adc >= ADC_SAT
    h["adc_anomalous"] = (h.adc.values > ADC_10BIT_MAX).astype(np.uint8)   # adc above the 10-bit maximum
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

    # cluster -> hit propagation (idmask, recovered; a per-hit rank is derived on the fly, not persisted)
    h = h.merge(c[["cl_id", "idmask", "recovered", "ontrack", "clean", "ntrk_share", "seedID"]],
                left_on="cid", right_on="cl_id", how="left").drop(columns="cl_id")
    h["idmask"] = h.idmask.fillna(0).astype(np.int64)
    h["recovered"] = h.recovered.fillna(0).astype(np.uint8)
    h["ontrack"] = h.ontrack.fillna(False).astype(bool)
    h["clean"] = h.clean.fillna(False).astype(bool)
    h["ntrk_share"] = h.ntrk_share.fillna(0).astype(np.int64)
    h["seedID"] = h.seedID.fillna(-1).astype(np.int64)
    return h, c

def review_table(h, c):
    """per hitset: how much needs a human. n_notfull_cl selects on recovered, not idmask==15."""
    g = h.groupby("hs")
    key = g[["event", "layer", "phielem", "zelem"]].first()
    r = pd.DataFrame({
        "n_hits": g.size(),
        "n_notfull_cl": c[c.recovered != 1].groupby("hs").size(),
        "n_split_comp": h[h.comp_ncl >= 2].drop_duplicates(["hs", "comp"]).groupby("hs").size(),
        "n_orphan_hits": h[(h.cid < 0) & (h.comp_n >= 3) & ~h.subthr].groupby("hs").size(),
        "n_ontrack_cl": c[c.ontrack].groupby("hs").size(),
    }).fillna(0).astype(int)
    r["score"] = 3 * r.n_notfull_cl + 2 * r.n_split_comp + r.n_orphan_hits
    return key.join(r)

# ----------------------------------------------------------------------------- diagnostics
def idmask_crosstab(c):
    sub = c[c.idmask != 15]
    if not len(sub):
        return pd.DataFrame()
    conds = pd.DataFrame({
        "pedge_gt0": (sub.pedge > 0).values,
        "island_ge2cl": (sub.comp_ncl >= 2).values,
        "nhit0": (sub.nhit == 0).values,
        "tbin_m1": (sub.tbin == -1).values,
        "dup_entry": sub.duplicated(subset=DUP_COLS, keep=False).values,
        "nonfinite_pos": (~sub.finite).values,
    }, index=sub.index)
    ct = conds.groupby(sub.idmask.values).sum()
    ct["n_clusters"] = sub.groupby(sub.idmask.values).size()
    ct.index.name = "idmask"
    return ct

def bit4_fail_zero_adc(h, c):
    zero_adc_cids = set(h.loc[h.adc == 0, "cid"].unique()) - {-1}
    fail_b4 = c[(c.idmask.values & 4) == 0]
    return int(fail_b4.cl_id.isin(zero_adc_cids).sum()), int(len(fail_b4))

def offset_table(c):
    ok = (c.idmask.values & 1) > 0
    sub = c[ok]
    if not len(sub):
        return pd.DataFrame()
    g = sub.groupby(["layer", "phielem", "zelem"])
    return pd.DataFrame({
        "dt_median": g.dt.median(), "dt_q16": g.dt.quantile(0.16), "dt_q84": g.dt.quantile(0.84),
        "dp_median": g.dp.median(), "dp_q16": g.dp.quantile(0.16), "dp_q84": g.dp.quantile(0.84),
        "n": g.size(),
    }).reset_index()

# ----------------------------------------------------------------------------- one event, full pipeline
def process_event(f, starts, call, dead, e, out, toff, poff):
    h = read_event_hits(f, starts, e)
    c = call[call.event == e].reset_index(drop=True).copy()
    if len(h) == 0:
        return None, None
    h, c = label_event(h, c, dead, toff, poff)

    write_root_tree_fixed(f"{out}/hits_ev{e:03d}.root", "hits", h, HITS_SCHEMA)
    write_root_tree_fixed(f"{out}/clusters_ev{e:03d}.root", "clusters", c, CLUSTERS_SCHEMA)
    ot = offset_table(c)
    ot.to_csv(f"{out}/offsets_ev{e:03d}.csv", index=False)

    rv = review_table(h, c)

    laser = int(e == LASER_EVENT)
    ok = (c.idmask.values & 1) > 0
    frac_idmask15 = float(np.mean(c.idmask.values == 15)) if len(c) else float("nan")
    frac_recovered = float(np.mean(c.recovered.values == 1)) if len(c) else float("nan")
    frac_assigned = float(np.mean(h.cid.values >= 0)) if len(h) else float("nan")
    full = h.recovered.values == 1
    frac_clean_ontrack_full = float(np.mean((h.cid.values >= 0) & full & h.ontrack.values & h.clean.values)) if len(h) else float("nan")
    dt_med = float(np.nanmedian(c.dt.values[ok])) if ok.any() else float("nan")
    dp_med = float(np.nanmedian(c.dp.values[ok])) if ok.any() else float("nan")
    unassigned = h.cid.values < 0
    frac_small_comp = float(np.mean(h.comp_n.values[unassigned] <= 2)) if unassigned.any() else float("nan")
    frac_subthr = float(np.mean(h.subthr.values[unassigned])) if unassigned.any() else float("nan")

    # per-event residual: fixed toff/poff, bits-1,2,4-passing clusters with a valid (non-sentinel) tbin
    sel124 = (c.idmask.values & 7) == 7
    valid_tbin = c.tbin_sentinel.values == 0
    dt_resid_sample = c.dt.values[sel124 & valid_tbin]
    dt_median_resid = float(np.nanmedian(dt_resid_sample)) if len(dt_resid_sample) else float("nan")

    # event-level quality columns
    tbin_q50 = float(np.quantile(h.tbin.values, 0.50)) if len(h) else float("nan")
    tbin_q99 = float(np.quantile(h.tbin.values, 0.99)) if len(h) else float("nan")
    fail_id = (c.idmask.values & 7) != 7
    single = c.comp_ncl.values == 1
    multi = c.comp_ncl.values >= 2
    n_single, n_multi = int(single.sum()), int(multi.sum())
    n_fail_single, n_fail_multi = int((fail_id & single).sum()), int((fail_id & multi).sum())
    notrec = c[c.recovered.values == 0]
    frac_fail_streak = (float(np.mean((notrec.zsize.values >= STREAK_ZSIZE) | (notrec.phisize.values >= STREAK_PHISIZE)))
                         if len(notrec) else float("nan"))

    summary = dict(
        event=e, n_hits=len(h), n_clusters=len(c), max_tbin=int(h.tbin.max()) if len(h) else -1,
        tbin_q50=tbin_q50, tbin_q99=tbin_q99,
        frac_idmask15=frac_idmask15, frac_recovered=frac_recovered,
        frac_assigned=frac_assigned, frac_clean_ontrack_full=frac_clean_ontrack_full,
        dt_median=dt_med, dp_median=dp_med, dt_median_resid=dt_median_resid,
        n_single=n_single, n_fail_single=n_fail_single, n_multi=n_multi, n_fail_multi=n_fail_multi,
        frac_fail_streak=frac_fail_streak,
        frac_unassigned_small_comp=frac_small_comp, frac_unassigned_subthr=frac_subthr,
        laser_event=laser,
    )
    print(f"ev {e:3d}: hits {len(h):7d} cl {len(c):6d} idmask15 {frac_idmask15:.4f} recovered {frac_recovered:.4f} "
          f"assigned {frac_assigned:.4f} clean_ontrack_full {frac_clean_ontrack_full:.4f} | "
          f"dt_med {dt_med:+.3f} dp_med {dp_med:+.3f} dt_med_resid {dt_median_resid:+.3f} | "
          f"tbin_q50 {tbin_q50:.0f} tbin_q99 {tbin_q99:.0f} | laser={laser}")
    return summary, rv


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=os.environ.get("REAL_NTUPLE")); ap.add_argument("--events", default="all")
    ap.add_argument("--out", default=LABELS_DIR)
    ap.add_argument("--toff", type=float, default=FIXED_TOFF)      # fixed for the whole run
    ap.add_argument("--poff", type=float, default=FIXED_POFF)
    ap.add_argument("--gate-check", action="store_true",
                     help="run event 1 only, print frac_recovered, do not write event_summary/review_hitsets")
    a = ap.parse_args()
    if not a.file:
        ap.error("no ntuple given: pass FILE or set REAL_NTUPLE")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    f = uproot.open(a.file)
    starts, nev = event_bounds(f)
    evs = range(1, nev + 1) if a.events == "all" else range(int(a.events.split("-")[0]), int(a.events.split("-")[-1]) + 1)

    table = build_pad_table(f); dead = dead_map(table)
    print(f"pad table: {len(table)} hitsets, dead pads: {sum(d.sum() for d in dead.values())}   [{time.time()-t0:.0f}s]")
    call = add_pad_centroid(load_clusters(f), table)
    print(f"clusters: {len(call)} TPC, on-track {call.ontrack.sum()}, clean {call.clean.sum()}   [{time.time()-t0:.0f}s]")

    if a.gate_check:
        summary, _ = process_event(f, starts, call, dead, 1, a.out, a.toff, a.poff)
        print(f"\nGATE G1' (event-1 frac_recovered >= 0.85): frac_recovered={summary['frac_recovered']:.4f} "
              f"-> {'PASS' if summary['frac_recovered'] >= 0.85 else 'FAIL'}")
        return

    reviews, summary_rows = [], []
    resid_flag_events = []
    for e in evs:
        summary, rv = process_event(f, starts, call, dead, e, a.out, a.toff, a.poff)
        if summary is None:
            continue
        summary_rows.append(summary)
        reviews.append(rv)
        if np.isfinite(summary["dt_median_resid"]) and abs(summary["dt_median_resid"]) > 0.05:
            resid_flag_events.append((e, summary["dt_median_resid"]))

    pd.DataFrame(summary_rows).to_csv(f"{a.out}/event_summary.csv", index=False)
    rv_all = pd.concat(reviews).sort_values("score", ascending=False) if reviews else pd.DataFrame()
    rv_all.to_csv(f"{a.out}/review_hitsets.csv", index=False)
    print(f"\n{len(summary_rows)} events written. dt_median_resid |.|>0.05 bin in {len(resid_flag_events)} events.")
    print(f"total wall time [{time.time()-t0:.0f}s]")

    with open(f"{LABELS_DIR}/RESULTS_LEDGER.txt", "a") as fh:
        fh.write(f"\n=== Phase 2a: label_tpc_hits.py run (events={a.events}, toff={a.toff}, poff={a.poff}) ===\n")
        fh.write(f"events written: {len(summary_rows)}; wall time {time.time()-t0:.0f}s\n")
        fh.write(f"events with |dt_median_resid| > 0.05 bin (event, value): {resid_flag_events}\n")

if __name__ == "__main__":
    main()
