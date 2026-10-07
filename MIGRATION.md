# Migrating 3D_ClusterFinding to a new machine

Written 2026-10-07 for the move from the WSL2/Linux workstation (NVIDIA GPU) to an Apple Silicon Mac.
Everything large travels in one folder, the *bundle* (`migration_2026-10-07/`); this file explains how to
use it. Files kept out of git travel in the bundle's `private/` area.

## 1. What this project is now

This repository is the ML-making side of the sPHENIX TPC cluster-finding work: labelling, dataset
preparation and model prototyping. For the real TPC ntuple (RUN_REF) it holds a first real-data product:
seven scripts in `GroundTruth_Making/` (`label_tpc_hits.py`, `root_io_utils.py`, `cluster_descriptors.py`,
`hotpad_census.py`, `make_plots.py`, `hitset_viewer.py`, `phase3_sim.py`) recover which hits belong to each
official cluster (certified by exact ADC-sum, max-ADC, bounding-box and centroid identities; 0.88 of
clusters over 100 events), attach the tracker's on-track flags, compute 11 cluster descriptors and census
the hot pads. The product is the directory `data/labels_v0`, which is never committed.

Simulated training data come from the TPC_Sim_Pipeline exports, interface version v6.2 (v6.3 is in
preparation). The layout of the exports is documented in that repository's README; the two relevant
sections are copied into the bundle (`data/sim_v62/interface_note_pipeline_README_2026-10-05.md`).

## 2. Bundle layout

Hard copies only (no symbolic links); every file is listed with its md5 in `MANIFEST.md5`. Sizes in MiB
unless stated.

```
migration_2026-10-07/
├── README_MIGRATION.md        this guide, plus a map of the files kept out of git
├── MANIFEST.md5               md5 of every file below (paths relative to this folder)
├── MANIFEST_sizes.txt         byte sizes (du -ab)
├── data/
│   ├── real/                  clusters_seeds_island_RUN_REF-0.root_ntuplizer.root          440
│   ├── real_reference/        real_complete61_hits.root 101, island_real_complete.root 45,
│   │                          real_deadpads.txt, real_single_island_pads.txt, real_hotpads.txt   (147 in all)
│   ├── labels_v0/             the real-data labelling product, complete                    448
│   └── sim_v62/               hit69_frames_production_v62.root 745, island91_frames_production_v62.root 669,
│                              prodclus_v62.root 235, digi_frames_production_v62.root 590,
│                              production_manifest_v62_block.txt,
│                              interface_note_pipeline_README_2026-10-05.md        (2.2 GiB in all)
├── reference/pipeline_code/   read-only copies of the pipeline's exporters and shared header   (0.1)
├── private/                   files kept out of git                                        (67)
├── env/                       records of the old Linux environment (conda export, pip freeze,
│                              versions used) and environment-macos-arm64.yml
└── repo/3D_ClusterFinding.bundle    this repository with its full history (git bundle)
```

The bundle is about 3.3 GiB. Not copied: `frames_pp_production_v62.root` (3.3 GiB) and
`island_frames_v62.root` (200 MiB); both stay in the pipeline's `island_post/` on the old machine, and their
md5 are in `data/sim_v62/production_manifest_v62_block.txt`.

## 3. Set-up on the new machine

`BUNDLE` below is the path of the copied bundle folder. Run every script from the repository root: its
default paths are relative to it.

1. **Verify the copy.** On Linux: `cd "$BUNDLE" && md5sum -c MANIFEST.md5`. macOS has `md5`, not `md5sum`
   (use `md5`, not `shasum`: the manifest holds MD5 sums):

   ```
   cd "$BUNDLE"
   while read -r sum file; do [ "$(md5 -q "$file")" = "$sum" ] || echo "MISMATCH $file"; done < MANIFEST.md5
   ```

   No output means every file matches. (With Homebrew coreutils, `gmd5sum -c MANIFEST.md5` also works.)

2. **Clone the repository**, from GitHub or offline from the bundle:

   ```
   git clone https://github.com/FanxinSun/3D_ClusterFinding.git
   # offline: git clone "$BUNDLE/repo/3D_ClusterFinding.bundle" 3D_ClusterFinding
   #          git -C 3D_ClusterFinding remote set-url origin https://github.com/FanxinSun/3D_ClusterFinding.git
   ```

3. **The labelling product.** Create `data/` (git-ignored) and point `data/labels_v0` at the bundle copy:

   ```
   cd 3D_ClusterFinding && mkdir -p data
   ln -s "$BUNDLE/data/labels_v0" data/labels_v0     # or, without a link: export LABELS_DIR="$BUNDLE/data/labels_v0"
   ```

   Some scripts write into the labels directory (run ledger, plots). To keep the bundle copy verifiable, copy
   it (`cp -R`) instead of linking, or run those scripts as in section 4.

4. **The real ntuple**, under its RUN_REF name: keep `$BUNDLE/data/real/clusters_seeds_island_RUN_REF-0.root_ntuplizer.root`
   where it is, or place a copy beside the repository folder, and tell the scripts where it is:

   ```
   export REAL_NTUPLE="$BUNDLE/data/real/clusters_seeds_island_RUN_REF-0.root_ntuplizer.root"
   ```

5. **Simulation exports.** `export TPC_SIM_DIR="$BUNDLE/data/sim_v62"`. On the old machine this was the
   pipeline's `island_post/` directory.

6. **Environment.**

   ```
   conda env create -f "$BUNDLE/env/environment-macos-arm64.yml" && conda activate torch_env
   ```

## 4. Smoke test

From the repository root, with `REAL_NTUPLE` set as in step 4. `LABELS_DIR` points the run ledger and the
plots at the smoke-test folder, so the bundled `labels_v0` stays untouched.

```
LABELS_DIR=data/labels_v0_smoke python GroundTruth_Making/label_tpc_hits.py "$REAL_NTUPLE" \
    --events 1-1 --out data/labels_v0_smoke --toff 0.6961883752
```

The event line must read `recovered 0.8870`. Compare with the bundled product:

```
python - <<'EOF'
import pandas as pd
new = pd.read_csv("data/labels_v0_smoke/event_summary.csv").set_index("event").frac_recovered
ref = pd.read_csv("data/labels_v0/event_summary.csv").set_index("event").frac_recovered
print(f"event 1 frac_recovered: new {new[1]:.4f}  bundled {ref[1]:.4f}")     # both 0.8870
EOF
```

Then one plot script run on that output:

```
LABELS_DIR=data/labels_v0_smoke python GroundTruth_Making/cluster_descriptors.py
mkdir -p data/labels_v0_smoke/plots data/png_smoke
LABELS_DIR=data/labels_v0_smoke PNG_DIR=data/png_smoke python GroundTruth_Making/make_plots.py
```

It prints four lines, `P1: ... written` to `P4: ... written`, and leaves six PDFs in
`data/labels_v0_smoke/plots/` and the PNG copies in `data/png_smoke/`. On the old machine the whole test takes
under 10 seconds.

## 5. Known differences

- **No CUDA on the Mac.** torch uses the MPS backend (`torch.device("mps")`) or the CPU; code that selects a
  CUDA device needs changing. None of the seven scripts imports torch; it matters only for the notebooks and
  future model work.
- **ROOT files, not Parquet.** The old environment had no pyarrow, so every output is a ROOT file written with
  uproot (`root_io_utils.py` forces classic TTrees: uproot 5.7.4 writes RNTuples by default, which read back
  as awkward arrays instead of data frames). The new environment includes pyarrow, but nothing uses it yet.
  If a newer uproot behaves differently, the smoke test shows it; `pip install uproot==5.7.4` restores the
  version used here (`env/versions_used.txt`).
- **ROOT itself is optional.** The scripts need only uproot; ROOT (conda-forge builds exist for arm64) is
  needed only to run the C++ probes `probe2.C` and `probe_ntuplizer.C`.
- **`phase3_sim.py` is the v6.1-era comparison**: it still names `hit69_frames_production_v61.root` and
  `island91_frames_production_v61.root` inside `TPC_SIM_DIR`; the bundle carries v6.2 only, and the script has
  not been re-run on v6.2.
- **The old environment files are records.** `env/torch_env_wsl.yml` and `env/torch_env_wsl_pip.txt` hold
  Linux builds and build-machine paths; create the Mac environment from `environment-macos-arm64.yml`.
- **`data/` and the ntuple are never committed**: `.gitignore` excludes `data/`, `*.root`, `*.csv`, `*.pdf`.
