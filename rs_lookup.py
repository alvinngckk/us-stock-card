# -*- coding: utf-8 -*-
"""Relative Strength (RS) lookup vs stock-universe distribution in rs.json."""
from __future__ import annotations
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_WEBAPP_ROOT = Path(__file__).resolve().parent
DEFAULT_RS_PATH = os.environ.get(
    "RS_JSON_PATH",
    str(_WEBAPP_ROOT / "data" / "rs.json"),
)


@lru_cache(maxsize=4)
def load_rs_universe(path: str = DEFAULT_RS_PATH) -> Tuple[pd.DataFrame, str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Stock-only rows for ranking (exclude extra_index_etf if present)
    extra = set(data.get("extra_index_etf") or [])
    rows = [r for r in data["rows"] if r.get("t") not in extra]
    # Prefer n_stocks if present; else count stock rows
    n = int(data.get("n_stocks") or data.get("n") or len(rows))
    rdf = pd.DataFrame(rows)
    asof = str(data.get("asof") or "")
    return rdf, asof, n


def rs_percentile(ret, col: str, path: str = DEFAULT_RS_PATH) -> Optional[int]:
    """Rank an external return vs stock-universe distribution (scale 1–99)."""
    if ret is None or (isinstance(ret, float) and np.isnan(ret)):
        return None
    rdf, _, _ = load_rs_universe(path)
    if col not in rdf.columns or len(rdf) == 0:
        return None
    s = rdf[col].astype(float)
    n = len(s)
    below = int((s < float(ret)).sum())
    equal = int((s == float(ret)).sum())
    pct = (below + 0.5 * equal) / n
    return int(np.clip(round(pct * 98 + 1), 1, 99))


def rs_color(v: Optional[int]) -> str:
    if v is None:
        return "#111"
    if v >= 70:
        return "#006b45"
    if v <= 30:
        return "#8b0000"
    return "#222222"


def universe_meta(path: str = DEFAULT_RS_PATH) -> dict:
    _, asof, n = load_rs_universe(path)
    return {"asof": asof, "n": n, "path": path}


def find_cached_rs(ticker: str, path: str = DEFAULT_RS_PATH) -> Optional[dict]:
    """Optional: if ticker already in rs.json, return its rs1/rs3/rs6 (still recompute returns from live)."""
    rdf, _, _ = load_rs_universe(path)
    row = rdf[rdf.t == ticker]
    if len(row) == 0:
        # also check full file including ETF extras
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data["rows"]:
            if r.get("t") == ticker:
                return r
        return None
    r = row.iloc[0]
    return dict(t=ticker, rs1=int(r.rs1), rs3=int(r.rs3), rs6=int(r.rs6),
                r1=float(r.r1), r3=float(r.r3), r6=float(r.r6))
