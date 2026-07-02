from __future__ import annotations
import glob
import os
import re
import time
from datastore import DataStore

FOLDER   = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\sweep1920"  
GLOB     = "*.bmp"
DSN      = r"C:\Users\mu00129\Desktop\slmnew10\qsa.db"            # path related to personal laptop first, will fix later
DATA_ROOT = r"C:\Users\mu00129\Desktop\slmnew10"          

RUN_KIND   = "beam_sweep"      
SLM        = "PLUTO"
WAVELENGTH = 633.0
OPERATOR   = "NanoLab"
NOTES      = "1920-column black/white boundary sweep (beam-footprint mapping)"
CAMERA     = "FLIR Grasshopper"
EXPOSURE_US = 85.0

COMPUTE_STATS = True   
DEDUP         = True   
PROGRESS_EVERY = 100


def parse_index(filename: str) -> int | None:
    """Last integer in the name: Capture_sweep_0870.bmp -> 870."""
    nums = re.findall(r"\d+", os.path.basename(filename))
    return int(nums[-1]) if nums else None


def pattern_for(index: int | None):
    """Map the file index to (pattern_kind, gray_value, params) per RUN_KIND."""
    if RUN_KIND == "calibration":
        return "square", index, {"gray_value": index}
    if RUN_KIND == "beam_sweep":
        return "split", None, {"boundary_col": index}
    return "other", None, {"index": index}


def ingest(folder=FOLDER, dsn=DSN, data_root=DATA_ROOT):
    files = sorted(glob.glob(os.path.join(folder, GLOB)))
    if not files:
        print(f"No files matching {GLOB} in {folder}")
        return
    print(f"Found {len(files)} images in {folder}")
    store = DataStore(dsn, data_root=data_root if not str(dsn).startswith(("postgres", "dbname=")) else None)
    print("Backend:", "PostgreSQL" if store.pg else "SQLite", "->", dsn)

    seen_paths = set()
    if DEDUP:
        seen_paths = {r["image_path"] for r in store.query("SELECT image_path FROM captures")}

    run = store.start_run(kind=RUN_KIND, slm=SLM, name=os.path.basename(folder),
                          wavelength_nm=WAVELENGTH, operator=OPERATOR,
                          software_ver="ingest_folder.py", notes=NOTES)
    print(f"Created run #{run}")

    t0 = time.time()
    added = skipped = 0
    for i, path in enumerate(files):
        idx = parse_index(path)
        rel = store._relpath(path)
        if DEDUP and rel in seen_paths:
            skipped += 1
            continue
        seen_paths.add(rel)

        kind, gray, params = pattern_for(idx)
        pat = store.add_pattern(run, slm=SLM, kind=kind, gray_value=gray, params=params)
        store.add_capture(run, pat, path,
                          exposure_us=EXPOSURE_US if COMPUTE_STATS else None,
                          frame_index=idx, camera=CAMERA)
        added += 1
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"  {i+1}/{len(files)}  ({added} added, {skipped} skipped)")

    store.end_run(run, notes_append=f" | ingested {added} images")
    dt = time.time() - t0
    print(f"\nDone: {added} added, {skipped} skipped in {dt:.1f}s "
          f"({added/max(dt,1e-9):.0f} img/s)")
    verify(store, run)
    store.close()


def verify(store: DataStore, run: int):
    print("\n--- verification ---")
    n_cap = store.query("SELECT COUNT(*) AS n FROM captures WHERE run_id=?", [run])[0]["n"]
    n_pat = store.query("SELECT COUNT(*) AS n FROM patterns WHERE run_id=?", [run])[0]["n"]
    rng = store.query("SELECT MIN(frame_index) AS lo, MAX(frame_index) AS hi FROM captures WHERE run_id=?", [run])[0]
    print(f"run #{run}: {n_pat} patterns, {n_cap} captures, index range {rng['lo']}..{rng['hi']}")
    rows = store.query(
        """SELECT c.frame_index, c.image_path, c.width, c.height,
                  ROUND(c.mean_intensity,1) AS mean
           FROM captures c WHERE c.run_id=?
           ORDER BY c.frame_index""", [run])
    if not rows:
        print("  (no new captures in this run)")
        return
    print("first / middle / last rows:")
    for r in (rows[0], rows[len(rows)//2], rows[-1]):
        print(f"  idx {r['frame_index']:>4}  {r['image_path']}  "
              f"{r['width']}x{r['height']}  mean={r['mean']}")


if __name__ == "__main__":
    ingest()
