# -*- coding: utf-8 -*-
"""Stock card lookup website — FastAPI."""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from card_render import build_card, resolve_asof  # noqa: E402
from rs_lookup import universe_meta  # noqa: E402

app = FastAPI(title="美股查股卡", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

EXAMPLES = ["AAPL", "NVDA", "JNJ", "SPY", "QQQ", "TLT", "IWM"]
_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 300  # seconds


def _get_card(ticker: str) -> dict:
    key = ticker.strip().upper()
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < CACHE_TTL:
        return _CACHE[key][1]
    result = build_card(key)
    _CACHE[key] = (now, result)
    return result


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, t: str | None = None):
    meta = universe_meta()
    ctx = {
        "request": request,
        "examples": EXAMPLES,
        "asof_hint": resolve_asof(),
        "rs_asof": meta["asof"],
        "rs_n": meta["n"],
        "error": None,
        "summary": None,
        "ticker": (t or "").upper().strip(),
    }
    if t and t.strip():
        try:
            card = _get_card(t.strip())
            ctx["summary"] = card["summary"]
            ctx["ticker"] = card["summary"]["ticker"]
        except Exception as e:
            ctx["error"] = str(e)
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_page(request: Request, ticker: str):
    meta = universe_meta()
    ctx = {
        "request": request,
        "examples": EXAMPLES,
        "asof_hint": resolve_asof(),
        "rs_asof": meta["asof"],
        "rs_n": meta["n"],
        "error": None,
        "summary": None,
        "ticker": ticker.upper().strip(),
    }
    try:
        card = _get_card(ticker)
        ctx["summary"] = card["summary"]
        ctx["ticker"] = card["summary"]["ticker"]
    except Exception as e:
        ctx["error"] = str(e)
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/api/card/{ticker}.png")
async def card_png(ticker: str):
    try:
        card = _get_card(ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(content=card["png"], media_type="image/png",
                    headers={"Cache-Control": f"public, max-age={CACHE_TTL}"})


@app.get("/api/summary/{ticker}")
async def card_summary(ticker: str):
    try:
        card = _get_card(ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return card["summary"]


@app.get("/health")
async def health():
    return {"ok": True, "asof_hint": resolve_asof(), **universe_meta()}
