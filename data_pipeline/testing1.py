from __future__ import annotations
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image
    _HAVE_IMG = True
except Exception:
    _HAVE_IMG = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_pg(dsn: str) -> bool:
    return dsn.startswith(("postgres://", "postgresql://")) or dsn.startswith("dbname=")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            {ID},
    name          TEXT,
    kind          TEXT,
    slm           TEXT,
    wavelength_nm {REAL},
    laser_power   {REAL},
    operator      TEXT,
    software_ver  TEXT,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    notes         TEXT
);
CREATE TABLE IF NOT EXISTS patterns (
    id          {ID},
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    slm         TEXT,
    kind        TEXT,
    gray_value  INTEGER,
    params      {JSON},
    file_path   TEXT,
    sha256      TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS captures (
    id             {ID},
    run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    pattern_id     INTEGER REFERENCES patterns(id) ON DELETE SET NULL,
    image_path     TEXT NOT NULL,
    sha256         TEXT,
    captured_at    TEXT NOT NULL,
    exposure_us    {REAL},
    gain_db        {REAL},
    frame_index    INTEGER,
    camera         TEXT,
    width          INTEGER,
    height         INTEGER,
    mean_intensity {REAL},
    extra          {JSON}
);
CREATE TABLE IF NOT EXISTS analysis_results (
    id          {ID},
    capture_id  INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    kind        TEXT,
    value       {REAL},
    unit        TEXT,
    method      TEXT,
    params      {JSON},
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pat_run ON patterns(run_id);
CREATE INDEX IF NOT EXISTS ix_cap_run ON captures(run_id);
CREATE INDEX IF NOT EXISTS ix_cap_pat ON captures(pattern_id);
CREATE INDEX IF NOT EXISTS ix_res_cap ON analysis_results(capture_id);
"""

_TOKENS = {
    "pg":     dict(ID="BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY",
                   JSON="JSONB", REAL="DOUBLE PRECISION"),
    "sqlite": dict(ID="INTEGER PRIMARY KEY AUTOINCREMENT",
                   JSON="TEXT", REAL="REAL"),
}


class DataStore:
    def __init__(self, dsn: str, data_root: Optional[str] = None):
        self.pg = _is_pg(dsn)
        self.ph = "%s" if self.pg else "?"
        self.data_root = Path(data_root).resolve() if data_root else None

        if self.pg:
            import psycopg2
            self._pg = psycopg2
            self.conn = psycopg2.connect(dsn)
            schema = _SCHEMA.format(**_TOKENS["pg"])
            with self.conn.cursor() as cur:
                cur.execute(schema)          # psycopg2 runs multiple statements at once
            self.conn.commit()
        else:
            self.conn = sqlite3.connect(dsn)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(_SCHEMA.format(**_TOKENS["sqlite"]))
            self.conn.commit()

    # -- backend-neutral primitives ---------------------------------------- #
    def _x(self, sql: str) -> str:
        return sql.replace("?", self.ph) if self.pg else sql

    def _json(self, d: dict):
        if self.pg:
            from psycopg2.extras import Json
            return Json(d or {})
        return json.dumps(d or {})

    def _insert(self, sql: str, params: list) -> int:
        sql = self._x(sql)
        if self.pg:
            with self.conn.cursor() as cur:
                cur.execute(sql + " RETURNING id", params)
                rid = cur.fetchone()[0]
            self.conn.commit()
            return rid
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur.lastrowid

    def _exec(self, sql: str, params: list) -> None:
        sql = self._x(sql)
        if self.pg:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
        else:
            self.conn.execute(sql, params)
        self.conn.commit()

    def _relpath(self, p: str) -> str:
        p = Path(p).resolve()
        if self.data_root:
            try:
                return str(p.relative_to(self.data_root))
            except ValueError:
                pass
        return str(p)

    def _img_stats(self, p: str):
        if not _HAVE_IMG or not os.path.exists(p):
            return (None, None, None)
        a = np.asarray(Image.open(p).convert("L"))
        return (int(a.shape[1]), int(a.shape[0]), float(a.mean()))

    def start_run(self, kind, slm="", name="", wavelength_nm=None, laser_power=None,
                  operator="", software_ver="", notes="") -> int:
        return self._insert(
            """INSERT INTO runs(name,kind,slm,wavelength_nm,laser_power,operator,
                                software_ver,started_at,notes)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            [name, kind, slm, wavelength_nm, laser_power, operator,
             software_ver, _now(), notes])

    def end_run(self, run_id, notes_append="") -> None:
        row = self.query("SELECT notes FROM runs WHERE id=?", [run_id])[0]
        notes = (row["notes"] or "") + notes_append
        self._exec("UPDATE runs SET ended_at=?, notes=? WHERE id=?",
                   [_now(), notes, run_id])

    def add_pattern(self, run_id, slm, kind, gray_value=None, params=None,
                    file_path=None) -> int:
        sha = _sha256(Path(file_path)) if file_path and os.path.exists(file_path) else None
        return self._insert(
            """INSERT INTO patterns(run_id,slm,kind,gray_value,params,file_path,sha256,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            [run_id, slm, kind, gray_value, self._json(params),
             self._relpath(file_path) if file_path else None, sha, _now()])

    def add_capture(self, run_id, pattern_id, image_path, exposure_us=None,
                    gain_db=None, frame_index=None, camera="", captured_at=None,
                    extra=None) -> int:
        w, h, mean = self._img_stats(image_path)
        sha = _sha256(Path(image_path)) if os.path.exists(image_path) else None
        return self._insert(
            """INSERT INTO captures(run_id,pattern_id,image_path,sha256,captured_at,
                                    exposure_us,gain_db,frame_index,camera,width,height,
                                    mean_intensity,extra)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [run_id, pattern_id, self._relpath(image_path), sha,
             captured_at or _now(), exposure_us, gain_db, frame_index, camera,
             w, h, mean, self._json(extra)])

    def add_result(self, capture_id, kind, value, unit="", method="", params=None) -> int:
        return self._insert(
            """INSERT INTO analysis_results(capture_id,kind,value,unit,method,params,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            [capture_id, kind, value, unit, method, self._json(params), _now()])

    def query(self, sql, params=None):
        sql = self._x(sql)
        if self.pg:
            from psycopg2.extras import RealDictCursor
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params or [])
                return cur.fetchall()
        return self.conn.execute(sql, params or []).fetchall()

    def query_df(self, sql, params=None):
        import pandas as pd
        return pd.read_sql_query(self._x(sql), self.conn, params=params or [])

    def abspath(self, stored_path: str) -> str:
        p = Path(stored_path)
        if not p.is_absolute() and self.data_root:
            p = self.data_root / p
        return str(p)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import tempfile
    dsn = os.environ.get("QSA_DSN") or os.path.join(tempfile.mkdtemp(), "demo.db")
    root = None if _is_pg(dsn) else os.path.dirname(dsn)
    store = DataStore(dsn, data_root=root)
    print("backend:", "PostgreSQL" if store.pg else "SQLite")
    run = store.start_run(kind="calibration", slm="ERIS", wavelength_nm=633,
                          operator="demo", notes="self-test")
    for g in (0, 128, 255):
        pat = store.add_pattern(run, "ERIS", "square", gray_value=g,
                                params={"rect": [400, 680, 150, 450], "seed": 42})
        cap = store.add_capture(run, pat, f"Capture_gray_{g:03d}.bmp",
                                exposure_us=85.0, frame_index=g, camera="FLIR")
        store.add_result(cap, "phase_shift", value=g / 255 * 2, unit="pi",
                         method="find_phi")
    store.end_run(run)
    for r in store.query(
            """SELECT p.gray_value, r.value, r.unit FROM captures c
               JOIN patterns p ON p.id=c.pattern_id
               JOIN analysis_results r ON r.capture_id=c.id
               WHERE c.run_id=? ORDER BY p.gray_value""", [run]):
        print(f"  gray {r['gray_value']:3d}  phase {r['value']:.3f} {r['unit']}")
    store.close()
