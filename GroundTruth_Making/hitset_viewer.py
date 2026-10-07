#!/usr/bin/env python
"""
hitset_viewer.py -- P5 hitset panels: hits and official clusters of selected hitsets.

Per event (event 1 and the busiest complete event, tbin_q99 >= 200): the 6
highest-score hitsets plus 2 random fully-recovered hitsets, written into one
multi-page PDF (<out>/plots/hitsets.pdf) with PNG copies. Event 44 (laser) is
excluded from the "busiest" search: it has 0 clusters (GL1 calibration event),
so no idmask/recovered hitset panels exist for it anyway. The box colour is
keyed off `recovered` (recovered replaces idmask == 15 as the selection used
in plots).

Adapted from an earlier prototype (view_hitset), with these differences:
  - the track-extrapolation block (needs cluster `r`, not selected here) is dropped;
  - savefig (into a single multi-page PDF) instead of show;
  - cert -> idmask (renamed cluster/hit column).
"""
import os
import numpy as np
import pandas as pd
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

RNG_SEED = 20260921  # for the "2 random hitsets" pick -- reproducible
LABELS_DIR = os.environ.get("LABELS_DIR", "data/labels_v0")   # product directory; override on a new machine
# PNG copies of the panels go here (created on first use)
PNG_DIR = os.environ.get("PNG_DIR", "data/png")


def view_hitset(ax, h, c, event, layer, sector, side, pad=3):
    m = (h.layer == layer) & (h.phielem == sector) & (h.zelem == side); hh = h[m]
    cc = c[(c.layer == layer) & (c.phielem == sector) & (c.zelem == side)]
    p0, p1 = hh.phibin.min() - pad, hh.phibin.max() + pad
    t0, t1 = hh.tbin.min() - pad, hh.tbin.max() + pad
    img = np.full((p1 - p0 + 1, t1 - t0 + 1), np.nan)
    img[hh.phibin - p0, hh.tbin - t0] = hh.adc
    ax.imshow(img, origin="lower", aspect="auto", norm=LogNorm(vmin=10, vmax=1000),
              extent=[t0 - .5, t1 + .5, p0 - .5, p1 + .5], cmap="viridis")
    col = {1: "w"}                                                    # recovered white, others red
    for _, r in cc.iterrows():
        bp = round(r.pad_c - (r.phisize - 1) / 2); bt = round(r.tbin - (r.zsize - 1) / 2)
        ax.add_patch(mp.Rectangle((bt - .5, bp - .5), r.zsize, r.phisize, fill=False, lw=1.5,
                                   ec=col.get(int(r.recovered), "r"), ls="-" if r.ontrack else ":"))
        ax.plot(r.tbin, r.pad_c, "+", c="w" if r.ontrack else "orange", ms=8)
        if r.ontrack and np.isfinite(r.spt):
            ax.text(r.tbin, r.pad_c + 0.6, f"s{int(r.seedID)} pt{r.spt:.2f}", c="w", fontsize=7)
    ax.set(xlabel="tbin", ylabel="phibin [pad]",
           title=f"ev {event} L{layer} sec{sector} side{side}  "
                 f"white=recovered red=not-recovered solid=on-track dotted=off-track")


def panels_for_event(ev, out):
    h = uproot.open(f"{out}/hits_ev{ev:03d}.root")["hits"].arrays(library="pd")
    c = uproot.open(f"{out}/clusters_ev{ev:03d}.root")["clusters"].arrays(library="pd")
    rv = pd.read_csv(f"{out}/review_hitsets.csv")
    rv = rv[rv.event == ev]

    top6 = rv.sort_values("score", ascending=False).head(6)[["layer", "phielem", "zelem", "score"]]

    allrec = (c.groupby(["layer", "phielem", "zelem"])["recovered"]
                .agg(lambda s: bool(np.all(s.values == 1)) and len(s) > 0))
    candidates = allrec[allrec].index.to_list()
    rng = np.random.default_rng(RNG_SEED + ev)
    pick_idx = rng.choice(len(candidates), size=min(2, len(candidates)), replace=False)
    random2 = [candidates[i] for i in pick_idx]

    panels = [(int(r.layer), int(r.phielem), int(r.zelem), f"ev{ev} top-score #{i+1} score={r.score}")
              for i, (_, r) in enumerate(top6.iterrows())]
    panels += [(L, s, sd, f"ev{ev} random fully-recovered hitset") for (L, s, sd) in random2]
    return h, c, panels, random2


def main():
    import pandas as pd
    summary = pd.read_csv(f"{LABELS_DIR}/event_summary.csv")
    complete_nonlaser = summary[(summary.tbin_q99 >= 200) & (summary.laser_event == 0)]
    busiest = int(complete_nonlaser.sort_values("n_hits", ascending=False).iloc[0].event)
    print(f"busiest complete non-laser event: {busiest}")

    out = LABELS_DIR
    out_pdf = f"{out}/plots/hitsets.pdf"
    os.makedirs(PNG_DIR, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for ev in (1, busiest):
            h, c, panels, random2 = panels_for_event(ev, out)
            print(f"event {ev}: {len(panels)} panels; random2 (seed={RNG_SEED + ev}): {random2}")
            for L, s, sd, tag in panels:
                fig, ax = plt.subplots(figsize=(14, 6))
                view_hitset(ax, h, c, ev, L, s, sd)
                ax.text(0.01, 1.02, tag, transform=ax.transAxes, fontsize=8, color="gray")
                fig.tight_layout()
                pdf.savefig(fig)
                png_path = f"{PNG_DIR}/hitset_ev{ev}_L{L}_s{s}_sd{sd}.png"
                fig.savefig(png_path, dpi=110)
                plt.close(fig)
                print(f"  panel L{L} sec{s} side{sd}  [{tag}]")

    print(f"\n16 panels written -> {out_pdf}")


if __name__ == "__main__":
    main()
