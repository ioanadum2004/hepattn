from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

# Which ROOT files / trees to read (change PARTS_FILE if the match check below is low)
HITS_FILE, HITS_TREE = "measurements_with_noise.root", "measurements"
PARTS_FILE, PARTS_TREE = "particles_simulation.root", "particles"

ID_PARTS = ["vertex_primary", "vertex_secondary", "particle", "generation", "sub_particle"]


def pack_id(vp, vs, p, g, sp):
    """Combine the 5 ACTS ID numbers into ONE integer (same recipe for hits and particles)."""
    vp, vs, p, g, sp = (np.asarray(a, dtype=np.uint64) for a in (vp, vs, p, g, sp))
    packed = (vp << np.uint64(52)) | (vs << np.uint64(40)) | (p << np.uint64(24)) | (g << np.uint64(16)) | sp
    return packed.view(np.int64)


def first_or_zero(jagged):
    """Each hit has a list of particles (0, 1 or 2 long). Take the first, or 0 if empty."""
    return ak.to_numpy(ak.fill_none(ak.firsts(jagged), 0))


def preprocess(in_dir: str, out_dir: str, overwrite: bool):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read the hits table (all events at once, it is one flat table) ----
    hit_cols = ["event_nr", "volume_id", "true_x", "true_y", "true_z"] + [f"particles_{n}" for n in ID_PARTS]
    h = uproot.open(in_dir / HITS_FILE)[HITS_TREE].arrays(hit_cols, library="ak")

    has_particle = ak.to_numpy(ak.num(h["particles_particle"]) > 0)
    hits_all = pd.DataFrame(
        {
            "event_nr": ak.to_numpy(h["event_nr"]),
            "volume_id": ak.to_numpy(h["volume_id"]),
            "x": ak.to_numpy(h["true_x"]),
            "y": ak.to_numpy(h["true_y"]),
            "z": ak.to_numpy(h["true_z"]),
            "particle_id": pack_id(*[first_or_zero(h[f"particles_{n}"]) for n in ID_PARTS]),
            "has_particle": has_particle,
        }
    )

    # TEMPORARY: drop noise hits (empty particle list). Their true_x/y/z are 0.
    n_before = len(hits_all)
    hits_all = hits_all[hits_all["has_particle"]].drop(columns="has_particle")
    print(f"Dropped {n_before - len(hits_all)} of {n_before} hits with no particle (noise)")

    # ---- Read the particles table (one row in the tree = one event) ----
    part_cols = ["event_id", "px", "py", "pz", "q"] + ID_PARTS
    pa = uproot.open(in_dir / PARTS_FILE)[PARTS_TREE].arrays(part_cols, library="ak")
    entry_of_event = {int(e): i for i, e in enumerate(ak.to_numpy(pa["event_id"]))}

    for event, hits in tqdm(list(hits_all.groupby("event_nr"))):
        event = int(event)
        name = f"event{event:09d}"  # data.py reads the number after "event"
        hits_out, parts_out = out_dir / f"{name}-hits.parquet", out_dir / f"{name}-parts.parquet"
        if hits_out.exists() and parts_out.exists() and not overwrite:
            continue
        if event not in entry_of_event:
            tqdm.write(f"Skipping {name}: not in particles file")
            continue

        i = entry_of_event[event]
        parts = pd.DataFrame(
            {
                "particle_id": pack_id(*[ak.to_numpy(pa[n][i]) for n in ID_PARTS]),
                "px": ak.to_numpy(pa["px"][i]),
                "py": ak.to_numpy(pa["py"][i]),
                "pz": ak.to_numpy(pa["pz"][i]),
                "q": ak.to_numpy(pa["q"][i]),
            }
        )

        # Sanity check: how many hits point to a particle that exists in the particles table?
        frac = hits["particle_id"].isin(parts["particle_id"]).mean()
        tqdm.write(f"{name}: {len(hits)} hits, {len(parts)} particles, {frac:.1%} of hits matched to a particle")

        # reset_index matters: data.py needs unique row labels
        hits.drop(columns="event_nr").reset_index(drop=True).to_parquet(hits_out)
        parts.reset_index(drop=True).to_parquet(parts_out)


if __name__ == "__main__":
    ap = ArgumentParser(description="Prep ALICE 3 ROOT files into parquet")
    ap.add_argument("-i", "--in_dir", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    preprocess(a.in_dir, a.out_dir, a.overwrite)
