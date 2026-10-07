#!/usr/bin/env python
"""
make_plots.py [--out data/labels_v0]

Plots P1-P4: P1 descriptor distributions per region (clean on-track vs not
on-track), P2 two-pad clusters, P3 rms_t vs tbin, P4 recovery quality (idmask
histogram, centroid residuals, idmask == 15 per layer, frac_recovered per
event). recovered == 1 replaces idmask == 15 as the selection everywhere --
P1/P2/P3 are built from cluster_descriptors.root, which is already
recovered-only. P5 (hitset panels) is separate (hitset_viewer.py).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LASER_EVENT = 44
LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels_v0")   # product directory; override on a new machine
# PNG copies of the figures go here (created on first use)
PNG_DIR = os.environ.get("PNG_DIR", "data/png")


def save(pdf, fig, name):
    pdf.savefig(fig)
    os.makedirs(PNG_DIR, exist_ok=True)
    fig.savefig(f"{PNG_DIR}/{name}.png", dpi=100)
    plt.close(fig)


def overlay_hist(ax, a, b, label_a, label_b, xlabel, bins=50, log_y=False, rng=None):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if rng is None and (len(a) or len(b)):
        lo = np.nanmin(np.concatenate([a, b])) if len(a) and len(b) else (a.min() if len(a) else b.min())
        hi = np.nanmax(np.concatenate([a, b])) if len(a) and len(b) else (a.max() if len(a) else b.max())
        rng = (lo, hi) if hi > lo else (lo - 1, lo + 1)
    if len(a):
        ax.hist(a, bins=bins, range=rng, density=True, histtype="step", color="tab:blue", label=f"{label_a} (n={len(a)})")
    if len(b):
        ax.hist(b, bins=bins, range=rng, density=True, histtype="step", color="tab:red", label=f"{label_b} (n={len(b)})")
    ax.set_xlabel(xlabel); ax.set_ylabel("normalised density")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=6)


def make_p1(desc, out):
    quant_adc = ["adc_max", "adc_sum", "adc_min"]
    quant_other = ["rms_t", "rms_phi", "kurt_phi", "kurt_t", "skew_t", "skew_phi"]
    units = {"adc_max": "adc", "adc_sum": "adc", "adc_min": "adc",
             "rms_t": "tbin", "rms_phi": "pad", "cog_t": "tbin", "cog_phi": "pad"}
    # kurt_phi/kurt_t have a long tail from rare few-hit clusters (sample kurtosis blows up at
    # small n) that otherwise stretches the axis until the real peak is invisible; clip the
    # viewing range and use log-y like the adc panels. The tail itself is still in the data/report.
    clip_rng = {"kurt_phi": (-5, 30), "kurt_t": (-5, 30)}
    log_y_cols = set(quant_adc) | set(clip_rng)
    for reg in (1, 2, 3):
        sub = desc[desc.region == reg]
        a = sub[(sub.ontrack == 1) & (sub.clean == 1)]
        b = sub[sub.ontrack == 0]
        with PdfPages(f"{out}/plots/descriptors_R{reg}.pdf") as pdf:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            for ax, q in zip(axes.flat, quant_adc + quant_other):
                overlay_hist(ax, a[q].values, b[q].values, "clean on-track", "not on-track",
                              f"{q} [{units.get(q, '1')}]", log_y=(q in log_y_cols), rng=clip_rng.get(q))
            fig.suptitle(f"Region {reg}: descriptor distributions, recovered clusters")
            fig.tight_layout()
            save(pdf, fig, f"P1_R{reg}_page1_9panels")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            overlay_hist(axes[0], a["cog_t"].values, b["cog_t"].values, "clean on-track", "not on-track", "cog_t [tbin]")
            overlay_hist(axes[1], a["cog_phi"].values, b["cog_phi"].values, "clean on-track", "not on-track", "cog_phi [pad]")
            fig.suptitle(f"Region {reg}: cluster centroid position (cluster-local coords)")
            fig.tight_layout()
            save(pdf, fig, f"P1_R{reg}_page2_cog")
    print("P1: descriptors_R{1,2,3}.pdf written")


def make_p2(desc, out):
    sub = desc[desc.phisize == 2]
    a = sub[(sub.ontrack == 1) & (sub.clean == 1)]
    b = sub[sub.ontrack == 0]
    with PdfPages(f"{out}/plots/two_pad_clusters.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(7, 5))
        overlay_hist(ax, a["p2"].values, b["p2"].values, "clean on-track", "not on-track",
                     "p2 (charge fraction on lesser pad)", bins=40, rng=(0, 0.5))
        fig.tight_layout(); save(pdf, fig, "P2_page1_p2_hist")

        fig, ax = plt.subplots(figsize=(7, 5))
        m = np.isfinite(sub.p2.values) & np.isfinite(sub.kurt_phi.values)
        ax.hist2d(sub.p2.values[m], sub.kurt_phi.values[m], bins=[40, 60], range=[[0, 0.5], [-3, 15]],
                   cmap="viridis", norm=matplotlib.colors.LogNorm())
        pgrid = np.linspace(0.01, 0.49, 200)
        curve = (1 - 6 * pgrid * (1 - pgrid)) / (pgrid * (1 - pgrid))
        ax.plot(pgrid, curve, "r-", lw=1.5, label="(1-6p(1-p))/(p(1-p))")
        ax.set_xlabel("p2"); ax.set_ylabel("kurt_phi (excess)"); ax.legend(fontsize=8)
        fig.tight_layout(); save(pdf, fig, "P2_page2_kurt_vs_p2")
    print("P2: two_pad_clusters.pdf written")


def make_p3(desc, out):
    sel = desc[(desc.ontrack == 1) & (desc.clean == 1) & (desc.in_drift_window == 1)]
    edges = np.arange(0, 260, 25)
    with PdfPages(f"{out}/plots/rms_t_vs_tbin.pdf") as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        for ax, reg in zip(axes, (1, 2, 3)):
            sub = sel[sel.region == reg]
            centers, meds, q16, q84 = [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = sub[(sub.tbin_abs >= lo) & (sub.tbin_abs < hi)]["rms_t"]
                if len(s) < 5:
                    continue
                centers.append((lo + hi) / 2)
                meds.append(s.median()); q16.append(s.quantile(0.16)); q84.append(s.quantile(0.84))
            centers, meds, q16, q84 = map(np.array, (centers, meds, q16, q84))
            ax.plot(centers, meds, "-o", ms=3, color="tab:blue")
            ax.fill_between(centers, q16, q84, alpha=0.3, color="tab:blue")
            ax.set_xlabel("tbin"); ax.set_title(f"region {reg}")
        axes[0].set_ylabel("rms_t [tbin]  (median, 16/84 band)")
        fig.suptitle("Clean on-track recovered clusters, inside drift window")
        fig.tight_layout(); save(pdf, fig, "P3_page1_rms_t_vs_tbin")
    print("P3: rms_t_vs_tbin.pdf written")


def make_p4(out):
    parts = []
    for cp in sorted(glob.glob(f"{out}/clusters_ev*.root")):
        ev = int(cp.split("clusters_ev")[-1].split(".root")[0])
        if ev == LASER_EVENT:
            continue
        c = uproot.open(cp)["clusters"].arrays(["layer", "idmask", "dt", "dp"], library="pd")
        if len(c):
            parts.append(c)
    allc = pd.concat(parts, ignore_index=True)
    del parts
    summary = pd.read_csv(f"{out}/event_summary.csv")

    with PdfPages(f"{out}/plots/recovery_quality.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals, counts = np.unique(allc.idmask.values, return_counts=True)
        ax.bar(vals, counts, color="tab:blue")
        ax.set_xlabel("idmask (bits: 1 sum, 2 max, 4 bbox, 8 centroid)"); ax.set_ylabel("clusters")
        ax.set_yscale("log"); ax.set_xticks(range(16))
        fig.tight_layout(); save(pdf, fig, "P4_page1_idmask_hist")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, col, xlab in zip(axes, ("dt", "dp"), ("dt [tbin]", "dp [pad]")):
            v = allc[col].values; v = v[np.isfinite(v)]
            ax.hist(v, bins=80, color="tab:blue")
            ax.set_xlabel(xlab); ax.set_ylabel("clusters"); ax.set_yscale("log")
        fig.suptitle("Centroid residuals, all clusters (non-laser events)")
        fig.tight_layout(); save(pdf, fig, "P4_page2_dt_dp")

        fig, ax = plt.subplots(figsize=(9, 5))
        byL = allc.groupby("layer")["idmask"].apply(lambda s: float(np.mean(s.values == 15)))
        ax.plot(byL.index, byL.values, "-o", ms=3, color="tab:blue")
        ax.set_xlabel("layer"); ax.set_ylabel("fraction idmask == 15"); ax.set_ylim(0, 1)
        fig.tight_layout(); save(pdf, fig, "P4_page3_idmask15_per_layer")

        fig, ax = plt.subplots(figsize=(8, 5))
        trunc = summary.tbin_q99 < 200
        ax.scatter(summary.n_hits[~trunc], summary.frac_recovered[~trunc], marker="o", s=25,
                   color="tab:blue", label="complete (tbin_q99>=200)")
        ax.scatter(summary.n_hits[trunc], summary.frac_recovered[trunc], marker="^", s=35,
                   color="tab:orange", label="end-truncated (tbin_q99<200)")
        ax.axhline(0.85, color="gray", ls=":", lw=1, label="gate G1' = 0.85")
        ax.set_xlabel("n TPC hits (event)"); ax.set_ylabel("frac_recovered"); ax.legend(fontsize=8)
        fig.tight_layout(); save(pdf, fig, "P4_page4_frac_recovered_vs_nhits")
    print("P4: recovery_quality.pdf written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=LABELS_DIR)
    a = ap.parse_args()
    desc = uproot.open(f"{a.out}/cluster_descriptors.root")["desc"].arrays(library="pd")
    make_p1(desc, a.out)
    make_p2(desc, a.out)
    make_p3(desc, a.out)
    make_p4(a.out)


if __name__ == "__main__":
    main()
