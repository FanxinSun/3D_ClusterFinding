"""Shared ROOT I/O helpers for data/labels_v0 production.

uproot.recreate(path)[name] = {dict of arrays} writes an RNTuple by default in
this environment's uproot (5.7.4); reading such an RNTuple back with
library="pd" silently returns an awkward.highlevel.Array instead of a
pandas.DataFrame. mktree()+extend() forces a classic TTree, which round-trips
through library="pd" correctly and is what every other tree in the input file
already is.
"""
import numpy as np
import pandas as pd
import uproot


def downcast_for_root(df):
    """bool -> uint8 (flags); float64 -> float32; integer columns -> the
    smallest of int8/int16/int32/int64 that fits the column's actual values."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == bool:
            df[col] = s.astype(np.uint8)
        elif np.issubdtype(s.dtype, np.floating):
            df[col] = s.astype(np.float32)
        elif np.issubdtype(s.dtype, np.integer):
            df[col] = pd.to_numeric(s, downcast="integer")
    return df


def write_root_tree(path, treename, df):
    df = downcast_for_root(df)
    data = {c: df[c].to_numpy() for c in df.columns}
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as fo:
        fo.mktree(treename, {k: v.dtype for k, v in data.items()})
        fo[treename].extend(data)


# ----------------------------------------------------------------------------- fixed output schemas
# Fixed output schema so per-event files concatenate with consistent dtypes
# (replaces the per-event automatic downcast_for_root for the labelling output).
HITS_SCHEMA = {
    "event": np.int16, "layer": np.int8, "phielem": np.int8, "zelem": np.int8,
    "phibin": np.int16, "tbin": np.int16, "adc": np.int32, "ientry": np.int32,   # adc is int32 (not int16/uint16)
    "cid": np.int32, "comp": np.int32, "hs": np.int16,
    "comp_n": np.int32, "comp_ncl": np.int16, "idmask": np.uint8,
    "seedID": np.int16, "ntrk_share": np.int8,
    "in_drift_window": np.uint8, "subthr": np.uint8, "satur": np.uint8,
    "edgepad": np.uint8, "deadadj": np.uint8, "ontrack": np.uint8, "clean": np.uint8,
    "recovered": np.uint8, "adc_anomalous": np.uint8,   # adc_anomalous: adc > 1023
}

CLUSTERS_SCHEMA = {
    "event": np.int32, "layer": np.int32, "phielem": np.int32, "zelem": np.int32,
    "phi": np.float32, "tbin": np.float32, "adc": np.int32, "maxadc": np.int32,
    "phisize": np.int32, "zsize": np.int32, "pedge": np.int32,
    "x": np.float32, "y": np.float32, "z": np.float32,
    "cl_id": np.int64, "finite": np.uint8, "ix": np.int64, "iy": np.int64, "iz": np.int64,
    "ntrk_share": np.int32, "seedID": np.int32, "clean": np.uint8,
    "spt": np.float32, "seta": np.float32, "alpha": np.float32, "beta": np.float32,
    "resz": np.float32, "resphi": np.float32, "ontrack": np.uint8, "pad_c": np.float32,
    "idmask": np.uint8, "nhit": np.int32, "dt": np.float32, "dp": np.float32,
    "comp": np.int32, "hs": np.int32, "ncomp": np.int32, "comp_ncl": np.int32,
    "tbin_sentinel": np.uint8, "recovered": np.uint8,
}


def cast_fixed(df, schema, label):
    """Cast df to an exact {col: dtype} schema. Raises ValueError (does not
    silently widen) if any column is missing or its actual values don't fit
    the target integer range or overflow float32."""
    problems = []
    out = {}
    for col, dtype in schema.items():
        dt = np.dtype(dtype)
        if col not in df.columns:
            problems.append(f"{label}.{col}: column missing")
            continue
        v = df[col].to_numpy()
        if dt.kind in "iu":
            info = np.iinfo(dt)
            if len(v):
                if v.dtype.kind == "f" and np.isnan(v).any():
                    # nanmin/nanmax below would silently ignore these; an integer dtype cannot hold NaN
                    problems.append(f"{label}.{col}: {int(np.isnan(v).sum())} NaN value(s), cannot cast to {dt}")
                    continue
                vmin, vmax = np.nanmin(v), np.nanmax(v)
                if vmin < info.min or vmax > info.max:
                    problems.append(f"{label}.{col}: range [{vmin},{vmax}] does not fit {dt} [{info.min},{info.max}]")
                    continue
            out[col] = v.astype(dt)
        elif dt.kind == "f":
            v64 = v.astype(np.float64)
            v32 = v64.astype(np.float32)
            bad = np.isfinite(v64) & ~np.isfinite(v32)
            if bad.any():
                problems.append(f"{label}.{col}: {int(bad.sum())} finite values overflow float32")
                continue
            out[col] = v32
        else:
            problems.append(f"{label}.{col}: unsupported schema dtype {dt}")
    extra = [c for c in df.columns if c not in schema]
    if problems:
        raise ValueError(f"fixed-schema cast failed for {label} (extra columns ignored: {extra}):\n" + "\n".join(problems))
    return out


def write_root_tree_fixed(path, treename, df, schema):
    data = cast_fixed(df, schema, f"{treename}@{path}")
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as fo:
        fo.mktree(treename, {k: v.dtype for k, v in data.items()})
        fo[treename].extend(data)
