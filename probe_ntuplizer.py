#!/usr/bin/env python3
"""
probe_ntuplizer.py -- structural probe of a TpcClusterizer ntuplizer file (TPC-only where it matters)

usage:
  python probe_ntuplizer.py FILE_ntuplizer.root                 # heavy ntp_hit probes on the first 5M entries
  python probe_ntuplizer.py FILE.root --nhits 0                 # full ntp_hit for everything (slow: ~3 GB read)
  python probe_ntuplizer.py FILE.root --event 3 --no-pdf
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import uproot

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
try:
    from scipy import ndimage
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", "{:.5g}".format)

T0 = time.time()
TPC_LO, TPC_HI = 7, 54
NPAD_LAYER = (1152, 1536, 2304)      # nominal phibins per layer in R1/R2/R3
NPAD_SECTOR = (96, 128, 192)
ADC_SAT = 963                        # 1023 - 60 pedestal, seen as adc_max in all regions
QS = (1, 5, 16, 50, 84, 95, 99)


# ----------------------------------------------------------------------------- helpers
def banner(t):
    print(f"\n{'=' * 20} {t} {'=' * 20}   [{time.time() - T0:7.1f}s]", flush=True)


def region(layer):
    L = np.asarray(layer).astype(int)
    return np.select([L < TPC_LO, L <= 22, L <= 38, L <= TPC_HI], [-1, 0, 1, 2], -1)


def is_tpc(layer):
    return (layer >= TPC_LO) & (layer <= TPC_HI)


def quant(a, ps=QS):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "empty"
    return " ".join(f"q{p:02d}={v:.4g}" for p, v in zip(ps, np.quantile(a, np.array(ps) / 100)))


def numeric_branches(tree):
    one = tree.arrays(entry_stop=1, library="np")
    return [k for k, v in one.items() if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "iuf"]


def minmax_full(tree, branches, step):
    lo = {b: np.inf for b in branches}
    hi = {b: -np.inf for b in branches}
    nbad = {b: 0 for b in branches}
    for ch in tree.iterate(branches, library="np", step_size=step):
        for b in branches:
            v = ch[b]
            if v.dtype.kind == "f":
                m = np.isfinite(v)
                nbad[b] += int((~m).sum())
                v = v[m]
            if v.size:
                lo[b] = min(lo[b], float(v.min()))
                hi[b] = max(hi[b], float(v.max()))
    return lo, hi, nbad


def print_minmax(lo, hi, nbad):
    print(f"  {'branch':<12}{'min':>18}{'max':>18}   note")
    for b in lo:
        note = []
        if not np.isfinite(lo[b]):
            note.append("EMPTY / all non-finite")
        elif lo[b] == hi[b]:
            note.append("CONSTANT")
        if nbad[b]:
            note.append(f"{nbad[b]} non-finite")
        print(f"  {b:<12}{lo[b]:>18.7g}{hi[b]:>18.7g}   {' '.join(note)}")


def profile_fit(x, y, minn=20):
    """weighted straight-line fit to the per-integer-x mean of y (like fitting a TProfile)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    xi = np.rint(x[m]).astype(np.int64)
    off = xi.min()
    cnt = np.bincount(xi - off)
    sm = np.bincount(xi - off, weights=y[m])
    ok = cnt >= minn
    if ok.sum() < 2:
        return None
    xs = (np.nonzero(ok)[0] + off).astype(float)
    ys = sm[ok] / cnt[ok]
    p = np.polyfit(xs, ys, 1, w=np.sqrt(cnt[ok]))
    res = ys - np.polyval(p, xs)
    return dict(z0=p[1], slope=p[0], xmin=xs.min(), xmax=xs.max(), rms=np.sqrt(np.average(res ** 2, weights=cnt[ok])))


def report_fit(tag, r, yname="z"):
    if r is None:
        print(f"  {tag:<26}: not enough data")
        return
    print(f"  {tag:<26}: {yname} = {r['z0']:9.4f} + {r['slope']:9.5f}*tbin   tbin populated [{r['xmin']:.0f}, {r['xmax']:.0f}]"
          f"   {yname}(tbin=0)={r['z0']:.2f}   {yname}(tbin={r['xmax']:.0f})={r['z0'] + r['slope'] * r['xmax']:.2f}"
          f"   rms about line {r['rms']:.3f}")


def int8_wrap(v):
    return ((np.asarray(v).astype(np.int64) + 128) % 256) - 128


def coord_key(df):
    return pd.DataFrame({
        "event": df["event"].astype(np.int32).values,
        "ix": np.rint(df["x"].values.astype(np.float64) * 1e4).astype(np.int64),
        "iy": np.rint(df["y"].values.astype(np.float64) * 1e4).astype(np.int64),
        "iz": np.rint(df["z"].values.astype(np.float64) * 1e4).astype(np.int64),
    })


def get_tree(f, name):
    try:
        return f[name]
    except Exception:
        return None


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--nhits", type=int, default=5_000_000, help="ntp_hit entries loaded for heavy probes (0 = all)")
    ap.add_argument("--event", type=int, default=1, help="event used for the connected-component check")
    ap.add_argument("--pdf", default="probe_ntuplizer_py.pdf")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--step", default="256 MB", help="uproot chunk size for full-tree passes")
    args = ap.parse_args()
    make_pdf = HAVE_MPL and not args.no_pdf
    pdf = PdfPages(args.pdf) if make_pdf else None

    # ======================================================================= [0]
    banner("[0] file / trees")
    f = uproot.open(args.file)
    print(" ", args.file)
    for k in f.keys():
        print("  key", k, "  (uproot reads the highest cycle)" if k.endswith(";1") is False else "")
    ti, th, tc, tk = (get_tree(f, n) for n in ("ntp_info", "ntp_hit", "ntp_cluster", "ntp_clus_trk"))
    for n, t in (("ntp_info", ti), ("ntp_hit", th), ("ntp_cluster", tc), ("ntp_clus_trk", tk)):
        print(f"  {n:<14}{(t.num_entries if t is not None else 0):>12d} entries")
    if th is None or tc is None or ti is None:
        sys.exit("missing trees")
    if tk is not None:
        print(f"  ntp_clus_trk / ntp_cluster = {100. * tk.num_entries / tc.num_entries:.2f} %")
    NH = th.num_entries if args.nhits <= 0 else min(args.nhits, th.num_entries)
    print(f"  heavy ntp_hit probes use the first {NH} of {th.num_entries} entries")

    # ======================================================================= [1]
    banner("[1] ntp_info : columns, first 10 events, summaries, consistency with the other trees")
    info = ti.arrays(library="pd")
    print("  columns:", " ".join(info.columns))
    print(info.head(10).to_string())
    keep = [c for c in ("nhittpcall", "nhittpcin", "nhittpcmid", "nhittpcout", "nhitmvtx", "nhitintt", "nhittpot",
                        "nclusall", "nclustpc", "nclusmaps", "nclusintt", "nclusmms", "ntpcseed", "ntrk",
                        "occ11", "occ116", "occ21", "occ216", "occ31", "occ316") if c in info]
    print(info[keep].describe().T.to_string())
    S = {c: float(info[c].sum()) for c in keep}
    allhits = sum(S.get(c, 0) for c in ("nhittpcall", "nhitmvtx", "nhitintt", "nhittpot"))
    print(f"  sum(nhittpcall)                      = {S.get('nhittpcall', 0):.0f}")
    print(f"  sum(nhittpcall+mvtx+intt+tpot)       = {allhits:.0f}   vs ntp_hit entries {th.num_entries}   diff {th.num_entries - allhits:+.0f}")
    print(f"  sum(nclusall) = {S.get('nclusall', 0):.0f}  vs ntp_cluster entries {tc.num_entries}   ->"
          f" {'MATCH: ntp_cluster holds ALL detectors' if S.get('nclusall', 0) == tc.num_entries else 'mismatch'}")
    print(f"  sum(nclustpc) = {S.get('nclustpc', 0):.0f}  (expected number of TPC entries in ntp_cluster)")
    if tk is not None:
        print(f"  sum(ntrk) = {S.get('ntrk', 0):.0f}   sum(ntpcseed) = {S.get('ntpcseed', 0):.0f}   (compare with distinct (event,seedID) in [4])")
    if "nhittpcall" in info:
        cc = info[[c for c in keep if c.startswith(("nhit", "nclus", "ntpc", "ntrk", "occ"))]].corr().loc[["nhittpcall"]]
        print("  correlation of nhittpcall with the others:")
        print(cc.round(3).to_string())

    # ======================================================================= [2] hits
    banner("[2a] ntp_hit : first 25 entries (all detectors; layer 0-2 MVTX, 3-6 INTT, 7-54 TPC, 55-56 TPOT)")
    HCOLS = [c for c in ("event", "layer", "phielem", "zelem", "phibin", "zbin", "tbin", "adc", "e", "ecell", "z", "r", "phi")
             if c in th.keys()]
    hits = pd.DataFrame(th.arrays(HCOLS, library="np", entry_stop=NH))
    print(hits.head(25).to_string())

    banner("[2b] ntp_hit : min/max over the FULL tree, all detectors (CONSTANT = unfilled branch)")
    lo, hi, nbad = minmax_full(th, numeric_branches(th), args.step)
    print_minmax(lo, hi, nbad)

    tpc = hits[is_tpc(hits["layer"])].copy()
    tpc["reg"] = region(tpc["layer"])
    ev_loaded = hits["event"].astype(int).values
    last_ev = int(ev_loaded.max())
    complete_events = sorted(set(ev_loaded) - ({last_ev} if NH < th.num_entries else set()))
    print(f"\n  loaded entries {len(hits)}, TPC entries {len(tpc)} ({100. * len(tpc) / len(hits):.1f} %),"
          f" events {ev_loaded.min()}..{last_ev}, complete events in the loaded chunk: {len(complete_events)}")
    bad_side = ~tpc["zelem"].isin([0, 1])
    print(f"  TPC entries with zelem not in {{0,1}}: {int(bad_side.sum())}")
    tpc = tpc[~bad_side]

    banner("[2c] ntp_hit (TPC) : e vs adc, ecell vs adc")
    if "ecell" in tpc:
        print(f"  ecell == adc for all TPC entries: {bool((tpc['ecell'] == tpc['adc']).all())}")
    d = tpc["e"] - tpc["adc"]
    print(f"  e-adc in [{d.min():g}, {d.max():g}]   frac(e==adc)={float((d == 0).mean()):.4f}   e==0: {int((tpc['e'] == 0).sum())}"
          f"   e<adc: {int((d < 0).sum())}   e>0&&adc==0: {int(((tpc['e'] > 0) & (tpc['adc'] == 0)).sum())}")
    uns = tpc[(tpc["adc"] < ADC_SAT) & (tpc["e"] > 0)]
    if len(uns) > 1000:
        elo, ehi = np.quantile(uns["e"], [0.02, 0.98])
        m = uns[(uns["e"] > elo) & (uns["e"] < ehi)]
        p = np.polyfit(m["e"].astype(float), m["adc"].astype(float), 1)
        res = m["adc"] - np.polyval(p, m["e"])
        print(f"  adc = {p[1]:.3f} + {p[0]:.6f}*e  (fit {elo:.0f}<e<{ehi:.0f}, adc<{ADC_SAT})  -> {1. / p[0]:.2f} e per ADC count,"
              f" RMS(adc - fit) = {res.std():.3f} ADC  (flat RMS => e is pre-digitisation charge, RMS = noise)")
        print(f"  quantiles of e (TPC): {quant(tpc['e'])}")

    banner("[2d] ntp_hit (TPC) : phibin numbering per layer (global per layer expected: 1152/1536/2304)")
    for L in (7, 22, 23, 38, 39, 54):
        g = tpc[tpc["layer"] == L]
        if len(g) == 0:
            continue
        r = region(L)
        pb = g["phibin"].astype(int)
        distinct = pb.nunique()
        print(f"  layer {L:2d} : phibin in [{pb.min():5d}, {pb.max():5d}]   distinct {distinct:5d} / {NPAD_LAYER[r]}   n={len(g)}"
              f"   -> {'GLOBAL' if pb.max() > NPAD_SECTOR[r] else 'LOCAL (per sector)'}")
    for r, Ls in ((0, range(7, 23)), (1, range(23, 39)), (2, range(39, 55))):
        g = tpc[tpc["reg"] == r]
        if len(g) == 0:
            continue
        pb = g["phibin"].astype(int)
        nexp = NPAD_LAYER[r]
        missing = np.setdiff1d(np.arange(nexp), pb.unique())
        print(f"  R{r + 1}: distinct phibins {pb.nunique()} of {nexp}; missing ({len(missing)}): {missing[:48].tolist()}{' ...' if len(missing) > 48 else ''}")
        for div in (NPAD_SECTOR[r], NPAD_SECTOR[r] - 2):
            agree = float((pb // div == g["phielem"].astype(int)).mean())
            print(f"       phibin//{div} == phielem for {agree:.4f} of hits   (max phibin%{div} = {int((pb % div).max())})")

    banner("[2e] ntp_hit (TPC) : is zbin a function of (side, tbin) ?")
    for s in (0, 1):
        g = tpc[tpc["zelem"] == s]
        if len(g) == 0:
            continue
        nun = g.groupby("tbin")["zbin"].nunique()
        print(f"  side {s}: zbin in [{g['zbin'].min():g}, {g['zbin'].max():g}], {g['zbin'].nunique()} distinct values;"
              f" {len(nun)} tbin values, {int((nun > 1).sum())} with >1 distinct zbin ->"
              f" {'zbin IS a pure function of tbin' if (nun > 1).sum() == 0 else 'NOT a pure function of tbin'}")
        report_fit(f"  side {s} zbin vs tbin", profile_fit(g["tbin"], g["zbin"]), yname="zbin")
        print(f"       identical to tbin: {float((g['zbin'] == g['tbin']).mean()):.4f}   identical to round(z*?)... corr(zbin,z)={np.corrcoef(g['zbin'], g['z'])[0, 1]:+.4f}")

    banner("[2f] ntp_hit (TPC) : ADC spectrum per region -> zero-suppression threshold, saturation")
    adc_hist = {}
    for r in range(3):
        a = tpc.loc[tpc["reg"] == r, "adc"].astype(int).values
        if a.size == 0:
            continue
        h = np.bincount(np.clip(a, 0, 1023), minlength=1024)
        adc_hist[r] = h
        nz = np.nonzero(h)[0]
        thr, top = int(nz[0]), int(nz[-1])
        n = h.sum()
        print(f"  R{r + 1}: n={n}  adc_min={thr}  adc_max={top} ({h[top]} hits, {100. * h[top] / n:.3f} %)"
              f"  frac(adc<=thr+2)={h[:thr + 3].sum() / n:.3f}  (<=thr+5)={h[:thr + 6].sum() / n:.3f}  (<=thr+10)={h[:thr + 11].sum() / n:.3f}"
              f"  frac(adc<21)={h[:21].sum() / n:.4f}  frac(adc>=963)={h[963:].sum() / n:.4f}")
        print("       adc 0..40 :", " ".join(f"{i}:{h[i]}" for i in range(41)))
    low = tpc[tpc["adc"] < 21]
    if len(low):
        print(f"  hits with adc<21: {len(low)}  per layer:", dict(low.groupby("layer").size().astype(int)))
        print(f"     per sector:", dict(low.groupby("phielem").size().astype(int)))
        print("     examples:")
        print(low[["event", "layer", "phielem", "zelem", "phibin", "tbin", "adc", "e"]].head(12).to_string())
    else:
        print("  no TPC hit with adc<21")

    banner("[2g] ntp_hit (TPC) : z vs tbin per side  (z = z0 + slope*tbin from the file's own z column)")
    for s in (0, 1):
        g = tpc[tpc["zelem"] == s]
        if len(g) == 0:
            continue
        print(f"  side {s}: n={len(g)}  z in [{g['z'].min():.2f}, {g['z'].max():.2f}]  frac(z<0)={float((g['z'] < 0).mean()):.4f}"
              f"  tbin in [{g['tbin'].min():g}, {g['tbin'].max():g}]")
        report_fit(f"  side {s} z vs tbin", profile_fit(g["tbin"], g["z"]))
    print("  expectation: |z| = 105 cm at the readout (tbin=0) decreasing toward the CM, |slope| = vdrift*53ns ~ 0.42 cm/tbin")

    banner("[2h] ntp_hit (TPC) : <r> per layer, hits per region")
    rl = tpc.groupby("layer")["r"].mean()
    print("  " + " ".join(f"{int(L)}:{v:.2f}" for L, v in rl.items()))
    fr = tpc["reg"].value_counts(normalize=True).sort_index()
    print("  hits per region fraction:", " ".join(f"R{r + 1}={v:.3f}" for r, v in fr.items()))
    print(f"  phi range [{tpc['phi'].min():.4f}, {tpc['phi'].max():.4f}]")

    banner("[2i] ntp_hit : event ordering, hits/event, per-event check against ntp_info.nhittpcall")
    breaks = int((np.diff(ev_loaded) < 0).sum())
    print(f"  ordering breaks (event decreases) in loaded entries: {breaks} -> {'SORTED by event' if breaks == 0 else 'NOT sorted'}")
    per_ev_tpc = tpc.groupby("event").size()
    per_ev_all = hits.groupby("event").size()
    infoi = info.set_index(info["event"].astype(int))
    rows = []
    for e in complete_events[:8]:
        rows.append((e, int(per_ev_all.get(e, 0)), int(per_ev_tpc.get(e, 0)), int(infoi.loc[e, "nhittpcall"]) if e in infoi.index else -1))
    print("  event : entries(all det) : entries(TPC) : ntp_info.nhittpcall")
    for e, na, nt, ni in rows:
        print(f"    {e:3d} : {na:8d} : {nt:8d} : {ni:8d}   {'OK' if nt == ni else 'DIFF'}")
    if complete_events:
        ce = [e for e in complete_events if e in infoi.index]
        nmatch = sum(int(per_ev_tpc.get(e, 0)) == int(infoi.loc[e, "nhittpcall"]) for e in ce)
        print(f"  TPC entries per event == nhittpcall for {nmatch}/{len(ce)} complete events")
        m = per_ev_tpc.reindex(ce).fillna(0)
        print(f"  TPC hits/event (complete events): mean {m.mean():.0f} min {m.min():.0f} max {m.max():.0f}   -> {m.mean() / 1152:.1f} hits per hitset (48 layers x 12 sectors x 2 sides)")

    # ======================================================================= [3] clusters
    banner("[3a] ntp_cluster : first 15 entries (all detectors)")
    CCOLS = [c for c in ("event", "layer", "phielem", "zelem", "phibin", "tbin", "adc", "maxadc", "size", "phisize", "zsize",
                         "pedge", "fee", "chan", "sampa", "ez", "ephi", "e", "x", "y", "z", "r") if c in tc.keys()]
    clus = pd.DataFrame(tc.arrays(CCOLS, library="np"))
    print(clus.head(15).to_string())

    banner("[3b] ntp_cluster : min/max over the FULL tree, all detectors")
    lo, hi, nbad = minmax_full(tc, numeric_branches(tc), args.step)
    print_minmax(lo, hi, nbad)

    ctpc = clus[is_tpc(clus["layer"])].copy()
    ctpc["reg"] = region(ctpc["layer"])
    print(f"\n  TPC clusters: {len(ctpc)} of {len(clus)}  (sum nclustpc from ntp_info = {S.get('nclustpc', 0):.0f})")
    per_ev_c = ctpc.groupby("event").size()
    if "nclustpc" in info:
        ok = sum(int(per_ev_c.get(e, 0)) == int(infoi.loc[e, "nclustpc"]) for e in infoi.index)
        print(f"  TPC clusters per event == nclustpc for {ok}/{len(infoi)} events")

    banner("[3c] ntp_cluster (TPC) : centroid convention")
    fi_pb = float((ctpc["phibin"] == np.rint(ctpc["phibin"])).mean())
    fi_tb = float((ctpc["tbin"] == np.rint(ctpc["tbin"])).mean())
    print(f"  fraction with integer phibin: {fi_pb:.3f}   integer tbin: {fi_tb:.3f}   -> phibin is a {'bin index' if fi_pb > 0.99 else 'centroid'}, tbin is a {'bin index' if fi_tb > 0.99 else 'centroid'}")
    for L in (7, 22, 23, 38, 39, 54):
        g = ctpc[ctpc["layer"] == L]
        if len(g):
            print(f"  layer {L:2d} : cluster phibin in [{g['phibin'].min():g}, {g['phibin'].max():g}]  tbin in [{g['tbin'].min():.2f}, {g['tbin'].max():.2f}]  n={len(g)}")
    if "e" in ctpc:
        print(f"  cluster e == adc: {float((ctpc['e'] == ctpc['adc']).mean()):.4f}")

    banner("[3d] ntp_cluster (TPC) : size / shape / edge / electronics / z convention / duplicates")
    box = ctpc["phisize"].astype(int) * ctpc["zsize"].astype(int)
    print(f"  size == int8(phisize*zsize): {float((ctpc['size'].astype(int) == int8_wrap(box)).mean()):.5f}   size<0: {int((ctpc['size'] < 0).sum())}"
          f"   max phisize {ctpc['phisize'].max():g}  max zsize {ctpc['zsize'].max():g}  -> 'size' is the bounding box, not the hit count")
    print(f"  min adc {ctpc['adc'].min():g}   min maxadc {ctpc['maxadc'].min():g}   maxadc>={ADC_SAT}: {float((ctpc['maxadc'] >= ADC_SAT).mean()):.4f}")
    for r in range(3):
        g = ctpc[ctpc["reg"] == r]
        b = box[g.index]
        if len(g) == 0:
            continue
        print(f"  R{r + 1}: n={len(g):8d}  bbox=1 {float((b == 1).mean()):.4f}  =2 {float((b == 2).mean()):.4f}  =3 {float((b == 3).mean()):.4f}"
              f"  phisize==1 {float((g['phisize'] == 1).mean()):.4f}  zsize==1 {float((g['zsize'] == 1).mean()):.4f}"
              f"  zsize18-26 {float(g['zsize'].between(18, 26).mean()):.4f}  <ez> {g['ez'].mean():.4f} <ephi> {g['ephi'].mean():.5f}"
              f"  adc: {quant(g['adc'], (5, 50, 95))}")
    bump = ctpc[ctpc["zsize"].between(18, 26)]
    print(f"  zsize 18-26 population: {len(bump)} clusters, mean phisize {bump['phisize'].mean():.2f}, mean adc {bump['adc'].mean():.0f},"
          f" mean tbin {bump['tbin'].mean():.1f}; per layer: {dict(bump.groupby('layer').size().astype(int))}")
    print(f"  pedge>0 fraction {float((ctpc['pedge'] > 0).mean()):.4f}; distribution: {dict(ctpc['pedge'].astype(int).value_counts().sort_index())}")
    for c in ("fee", "chan", "sampa"):
        if c in ctpc:
            v = ctpc[c]
            fin = v[np.isfinite(v)]
            print(f"  {c}: finite {len(fin)}/{len(v)}  range [{fin.min() if len(fin) else np.nan:g}, {fin.max() if len(fin) else np.nan:g}]")
    for s in (0, 1):
        g = ctpc[ctpc["zelem"] == s]
        if len(g) == 0:
            continue
        print(f"  side {s}: n={len(g)}  <z>={g['z'].mean():.2f}  frac(z<0)={float((g['z'] < 0).mean()):.4f}  z in [{g['z'].min():.1f}, {g['z'].max():.1f}]")
        report_fit(f"  side {s} cluster z vs tbin", profile_fit(g["tbin"], g["z"]))
    print(f"  clusters per region fraction: {dict(ctpc['reg'].value_counts(normalize=True).sort_index().round(3))}")
    print(f"  size quantiles TPC: {quant(ctpc['size'])}")
    print(f"  phisize            : {quant(ctpc['phisize'])}")
    print(f"  zsize              : {quant(ctpc['zsize'])}")
    print(f"  adc                : {quant(ctpc['adc'])}")
    print(f"  maxadc             : {quant(ctpc['maxadc'])}")
    kc = coord_key(clus)
    dup = kc.duplicated(keep="first").values
    print(f"  exact duplicate (event,x,y,z) entries: TPC {int(dup[is_tpc(clus['layer']).values].sum())}   non-TPC {int(dup[~is_tpc(clus['layer']).values].sum())}")

    # ======================================================================= [4] clus_trk
    trk = None
    if tk is not None:
        banner("[4a] ntp_clus_trk : first 15 entries")
        trk = pd.DataFrame(tk.arrays(numeric_branches(tk), library="np"))
        show = [c for c in ("event", "seedID", "layer", "phibin", "tbin", "size", "sntpc", "snsil", "spt", "seta", "scharge", "sdedx",
                            "alpha", "beta", "resphi", "resz", "x", "y", "z") if c in trk]
        print(trk[show].head(15).to_string())

        banner("[4b] ntp_clus_trk : min/max (full tree)")
        lo, hi, nbad = minmax_full(tk, list(trk.columns), args.step)
        print_minmax(lo, hi, nbad)

        banner("[4c] ntp_clus_trk : quantiles, track bookkeeping, residual units, clean-track cut")
        ttpc = trk[is_tpc(trk["layer"])].copy()
        ttpc["reg"] = region(ttpc["layer"])
        print(f"  TPC entries {len(ttpc)} / {len(trk)}")
        for c in ("sntpc", "spt", "seta", "sdedx", "alpha", "beta", "resphi", "resz", "size"):
            if c in ttpc:
                print(f"  {c:<8}: {quant(ttpc[c])}")
        if "resphio" in ttpc:
            print(f"  resphio == resphi: {float((ttpc['resphio'] == ttpc['resphi']).mean()):.4f}")
        if "seedID" in ttpc:
            ng = trk.groupby(["event", "seedID"]).ngroups
            print(f"  distinct (event,seedID): {ng}   sum(ntrk)={S.get('ntrk', 0):.0f}   sum(ntpcseed)={S.get('ntpcseed', 0):.0f}"
                  f"   -> {'these are FINAL TRACKS' if ng == S.get('ntrk', -1) else 'these are TPC seeds' if ng == S.get('ntpcseed', -2) else 'neither'}"
                  f";  {len(trk) / max(ng, 1):.1f} clusters per track")
        if {"x", "y"} <= set(ttpc.columns):
            rad = np.hypot(ttpc["x"], ttpc["y"])
        else:
            rad = ttpc["layer"].map(rl).fillna(50.0)
        for r in range(3):
            m = ttpc["reg"] == r
            if m.sum() == 0:
                continue
            print(f"  R{r + 1}: RMS(resphi)={np.sqrt(np.mean(ttpc.loc[m, 'resphi'] ** 2)):.5f}"
                  f"   RMS(resphi*r)={np.sqrt(np.mean((ttpc.loc[m, 'resphi'] * rad[m]) ** 2)):.4f} cm"
                  f"   RMS(resz)={np.sqrt(np.mean(ttpc.loc[m, 'resz'] ** 2)):.4f} cm")
        print("  -> if RMS(resphi*r) ~ 0.05-0.2 cm and grows with r, resphi is in radians (RMS(resphi)=0.003 cm = 30 um would be unphysical)")
        if "zelem" in ttpc:
            side = ttpc["zelem"].astype(int)
            tag = "side from zelem"
        else:
            side = (ttpc["z"] > 0).astype(int)
            tag = "no zelem branch: 'side' = (z>0)"
        for s in (0, 1):
            g = ttpc[side == s]
            if len(g):
                report_fit(f"  {tag} {s}: z vs tbin", profile_fit(g["tbin"], g["z"]))
        need = {"sntpc", "spt", "seta", "resz"}
        if need <= set(ttpc.columns):
            clean = (ttpc["sntpc"] >= 30) & (ttpc["spt"] > 0.15) & (ttpc["spt"] < 10) & (ttpc["seta"].abs() < 1.1) & (ttpc["resz"].abs() < 1)
            print(f"  clean-track cut (sntpc>=30, 0.15<spt<10, |seta|<1.1, |resz|<1): {float(clean.mean()):.4f} of TPC entries")

        # =================================================================== [5] cross-tree
        banner("[5] cross-tree : ntp_clus_trk subset of ntp_cluster ? (exact join on event + x,y,z rounded to 1 um)")
        if {"x", "y", "z"} <= set(trk.columns):
            kt = coord_key(trk)
            kcu = kc.drop_duplicates().assign(found=1)
            mg = kt.merge(kcu, how="left", on=["event", "ix", "iy", "iz"])
            found = mg["found"].fillna(0).astype(bool)
            ndist = kt.drop_duplicates().shape[0]
            ntpc_dist = kc[is_tpc(clus["layer"]).values].drop_duplicates().shape[0]
            print(f"  ntp_cluster distinct keys {kcu.shape[0]} of {len(kc)}")
            print(f"  ntp_clus_trk entries found in ntp_cluster: {int(found.sum())} / {len(kt)} = {float(found.mean()):.4f}")
            print(f"  distinct clus_trk clusters {ndist}  (shared by >1 track: {len(kt) - ndist});  {100. * ndist / ntpc_dist:.2f} % of distinct TPC official clusters are on a track")
            print("  -> if ~1.0, on-track labels can be attached by exact (event,x,y,z) join")
        else:
            print("  ntp_clus_trk has no x,y,z -> cannot join")

    # ======================================================================= [6] connected components
    banner(f"[6] connected components of TPC hits in event {args.event} vs official clusters (per hitset = layer x sector x side)")
    if not HAVE_SCIPY:
        print("  scipy not available -> skipped")
    else:
        ev = args.event
        if ev in complete_events:
            hev = tpc[tpc["event"] == ev]
        else:
            print(f"  event {ev} not fully inside the loaded chunk -> reading it from the full tree")
            hev = pd.DataFrame(th.arrays(["event", "layer", "phielem", "zelem", "phibin", "tbin", "adc"], library="np",
                                          cut=f"(event=={ev})&(layer>={TPC_LO})&(layer<={TPC_HI})"))
            hev = hev[hev["zelem"].isin([0, 1])].copy()
            hev["reg"] = region(hev["layer"])
        if len(hev) == 0:
            print("  no TPC hits for this event")
        else:
            ncomp = {"4": np.zeros(3, int), "8": np.zeros(3, int)}
            sizes = {"4": [], "8": []}
            ndup = 0
            for (L, sc, s), g in hev.groupby(["layer", "phielem", "zelem"]):
                pb = g["phibin"].astype(int).values
                tb = g["tbin"].astype(int).values
                pb = pb - pb.min()
                tb = tb - tb.min()
                grid = np.zeros((pb.max() + 1, tb.max() + 1), bool)
                ndup += len(pb) - len(np.unique(pb * 100000 + tb))
                grid[pb, tb] = True
                r = region(L)
                for name, st in (("4", None), ("8", np.ones((3, 3), int))):
                    lab, n = ndimage.label(grid, structure=st)
                    ncomp[name][r] += n
                    if n:
                        sizes[name].append(np.bincount(lab.ravel())[1:])
            noff = ctpc[ctpc["event"] == ev].groupby("reg").size().reindex(range(3)).fillna(0).astype(int).values
            print(f"  TPC hits {len(hev)}  (duplicate (phibin,tbin) within a hitset: {ndup})")
            print(f"  {'':10}{'R1':>10}{'R2':>10}{'R3':>10}{'all':>10}")
            print(f"  {'official':<10}{noff[0]:>10d}{noff[1]:>10d}{noff[2]:>10d}{noff.sum():>10d}   (ntp_info.nclustpc = {int(infoi.loc[ev, 'nclustpc']) if ev in infoi.index else -1})")
            for name in ("4", "8"):
                c = ncomp[name]
                sz = np.concatenate(sizes[name]) if sizes[name] else np.array([0])
                print(f"  {name + '-conn':<10}{c[0]:>10d}{c[1]:>10d}{c[2]:>10d}{c.sum():>10d}   hits/comp {len(hev) / max(c.sum(), 1):.2f}"
                      f"  frac(1-hit comps) {float((sz == 1).mean()):.3f}  comp size {quant(sz, (50, 84, 95, 99))}")
            print("  -> official/components > 1 means the island clusterizer splits connected islands; < 1 means it merges or drops hits")

    # ======================================================================= plots
    if pdf is not None:
        banner(f"plots -> {args.pdf}")
        col = ("r", "g", "b")
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for r, h in adc_hist.items():
            ax[0, 0].step(np.arange(1024), h, color=col[r], label=f"R{r + 1}", where="mid")
            ax[0, 1].step(np.arange(1024)[:80], h[:80], color=col[r], label=f"R{r + 1}", where="mid")
        for a in ax[0]:
            a.set_yscale("log"); a.set_xlabel("hit adc"); a.legend()
        ax[0, 1].set_title("low-adc zoom (threshold)")
        for s in (0, 1):
            g = tpc[tpc["zelem"] == s]
            if len(g):
                ax[1, s].hist2d(g["tbin"], g["z"], bins=[200, 130], norm=LogNorm())
                ax[1, s].set_title(f"hit z vs tbin, side {s}"); ax[1, s].set_xlabel("tbin"); ax[1, s].set_ylabel("z [cm]")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for s in (0, 1):
            g = tpc[tpc["zelem"] == s]
            if len(g):
                ax[0, s].hist2d(g["tbin"], g["zbin"], bins=[200, 128], norm=LogNorm())
                ax[0, s].set_title(f"hit zbin vs tbin, side {s}")
            gc = ctpc[ctpc["zelem"] == s]
            if len(gc):
                ax[1, s].hist2d(gc["tbin"], gc["z"], bins=[200, 130], norm=LogNorm())
                ax[1, s].set_title(f"cluster z vs tbin, side {s}")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        m = tpc["e"] > 0
        ax[0, 0].hist2d(tpc.loc[m, "e"], tpc.loc[m, "adc"], bins=[200, 200], range=[[0, float(np.quantile(tpc.loc[m, "e"], 0.995))], [0, 1000]], norm=LogNorm())
        ax[0, 0].set_title("hit adc vs e (TPC)"); ax[0, 0].set_xlabel("e"); ax[0, 0].set_ylabel("adc")
        ax[0, 1].hist2d(ctpc["layer"], ctpc["zsize"].clip(upper=39), bins=[48, 40], range=[[6.5, 54.5], [-0.5, 39.5]], norm=LogNorm())
        ax[0, 1].set_title("cluster zsize vs layer"); ax[0, 1].set_xlabel("layer")
        ax[1, 0].hist2d(ctpc["tbin"], ctpc["zsize"].clip(upper=39), bins=[130, 40], range=[[-10, 380], [-0.5, 39.5]], norm=LogNorm())
        ax[1, 0].set_title("cluster zsize vs tbin"); ax[1, 0].set_xlabel("tbin")
        ax[1, 1].hist2d(ctpc["size"], box, bins=[128, 130], range=[[-128, 128], [0, 130]], norm=LogNorm())
        ax[1, 1].set_title("phisize*zsize vs size"); ax[1, 1].set_xlabel("size"); ax[1, 1].set_ylabel("phisize*zsize")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if trk is not None and {"x", "y", "resphi", "resz"} <= set(trk.columns):
            tt = trk[is_tpc(trk["layer"])]
            rr = np.hypot(tt["x"], tt["y"])
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax[0].hist2d(tt["layer"], tt["resphi"] * rr, bins=[48, 200], range=[[6.5, 54.5], [-0.5, 0.5]], norm=LogNorm())
            ax[0].set_title("resphi*r vs layer [cm if resphi in rad]")
            ax[1].hist2d(tt["layer"], tt["resz"], bins=[48, 200], range=[[6.5, 54.5], [-2, 2]], norm=LogNorm())
            ax[1].set_title("resz vs layer [cm]")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        pdf.close()
        print(f"  wrote {args.pdf}")

    banner("done")


if __name__ == "__main__":
    main()