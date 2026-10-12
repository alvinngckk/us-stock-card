# -*- coding: utf-8 -*-
"""Matplotlib stock/index card renderer (PDF-style) → PNG bytes."""
from __future__ import annotations
import io
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager as fm
import yfinance as yf

_WEBAPP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_WEBAPP_ROOT))
from dataio import load_df_official

from rs_lookup import rs_percentile, rs_color, universe_meta

FONT_PATH = os.environ.get(
    "CJK_FONT_PATH",
    str(_WEBAPP_ROOT / "fonts" / "wqy-microhei.ttf"),
)
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Noto Sans CJK HK", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ET = ZoneInfo("America/New_York")
HK = ZoneInfo("Asia/Hong_Kong")

# Known index/ETF tickers → Traditional Chinese title
KNOWN_INDEX = {
    "SPY": "標普500 ETF",
    "QQQ": "納指100 ETF",
    "IWM": "羅素2000 ETF",
    "TLT": "20年期以上美債 ETF",
    "RSP": "標普500等權 ETF",
    "MDY": "標普中型400 ETF",
    "DIA": "道指 ETF",
    "DX-Y.NYB": "美元指數（ICE DXY）",
    "^GSPC": "標普500指數",
    "^IXIC": "納斯達克綜合指數",
    "^DJI": "道瓊斯工業平均",
    "^RUT": "羅素2000指數",
    "^VIX": "VIX 波動率指數",
    "XLV": "醫療保健 ETF（SPDR）",
    "XLK": "科技 ETF（SPDR）",
    "XLF": "金融 ETF（SPDR）",
    "XLE": "能源 ETF（SPDR）",
    "XLI": "工業 ETF（SPDR）",
    "XLY": "非必需消費 ETF（SPDR）",
    "XLP": "必需消費 ETF（SPDR）",
    "XLU": "公用事業 ETF（SPDR）",
    "XLB": "原材料 ETF（SPDR）",
    "XLRE": "房地產 ETF（SPDR）",
    "XLC": "通訊服務 ETF（SPDR）",
    "IGV": "軟件 ETF（iShares）",
    "XBI": "生技 ETF（SPDR）",
    "XOP": "石油天然氣開採 ETF",
    "GLD": "黃金 ETF",
    "SLV": "白銀 ETF",
    "USO": "原油 ETF",
    "HYG": "高收益債 ETF",
    "LQD": "投資級債 ETF",
    "EEM": "新興市場 ETF",
    "EFA": "已發展市場 ETF",
    "ARKK": "ARK 創新 ETF",
    "SMH": "半導體 ETF",
    "SOXX": "半導體 ETF（iShares）",
}


def resolve_asof() -> str:
    """Last completed US cash-session calendar date (ET)."""
    now = datetime.now(ET)
    d = now.date()
    # Before ~16:00 ET on a weekday, prefer previous session
    if now.weekday() < 5 and now.hour < 16:
        d = d - timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun
        d = d - timedelta(days=1)
    return d.isoformat()


def _vol_fmt(v, pos=None):
    if v is None or (isinstance(v, float) and (math.isnan(v) or v < 0)):
        return ""
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.0f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def calc_rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def calc_kdj(df, n=9, m1=3, m2=3):
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def annotate_rsi(ax, rsi_series):
    val = float(rsi_series.dropna().iloc[-1])
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([val])
    ax2.set_yticklabels([f"{val:.1f}"])
    ax2.tick_params(axis="y", labelsize=9, colors="#0b6e4f", length=5, width=1.0, pad=2)
    for side in ("top", "bottom", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#16a085")
    ax2.spines["right"].set_linewidth(0.8)
    ax.axhline(val, color="#16a085", lw=0.5, ls=":", alpha=0.5, zorder=1)


def fmt_vol(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or v <= 0)):
        return "—"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def wrap_chars(text, width):
    lines, cur, n = [], "", 0
    for ch in text:
        cur += ch
        n += 1
        if n >= width:
            lines.append(cur)
            cur = ""
            n = 0
    if cur:
        lines.append(cur)
    return lines


def fmt_mcap(v):
    if not v:
        return "—"
    if v >= 1e12:
        return f"約 {v/1e12:.2f} 萬億美元"
    if v >= 1e8:
        return f"約 {v/1e8:,.0f} 億美元"
    if v >= 1e6:
        return f"約 {v/1e6:,.0f} 百萬美元"
    return f"{v:,.0f}"


def _ticker_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def detect_kind(ticker: str, info: Optional[dict] = None) -> str:
    t = ticker.upper()
    if t in KNOWN_INDEX or t.startswith("^"):
        return "index"
    info = info if info is not None else _ticker_info(ticker)
    qt = (info.get("quoteType") or "").upper()
    if qt in ("ETF", "INDEX", "MUTUALFUND", "CURRENCY", "FUTURE"):
        return "index"
    return "stock"


def next_earnings_label(ticker: str, info: Optional[dict] = None) -> str:
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            ed = cal["Earnings Date"]
            if isinstance(ed, (list, tuple)) and ed:
                d = ed[0]
            else:
                d = ed
            if hasattr(d, "strftime"):
                return f"下次業績：{d.strftime('%m/%d')}"
            return f"下次業績：{d}"
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # newer yfinance shape
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"]
                if hasattr(val, "iloc"):
                    val = val.iloc[0]
                if hasattr(val, "strftime"):
                    return f"下次業績：{val.strftime('%m/%d')}"
        # earnings_dates
        try:
            edf = t.get_earnings_dates(limit=4)
            if edf is not None and len(edf):
                now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
                idx = edf.index
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert("America/New_York").tz_localize(None)
                future = [d for d in idx if d >= now - pd.Timedelta(days=1)]
                if future:
                    return f"下次業績：{future[0].strftime('%m/%d')}"
        except Exception:
            pass
    except Exception:
        pass
    return ""


def compute_metrics(ticker: str, asof: Optional[str] = None, spy_df: Optional[pd.DataFrame] = None):
    asof = asof or resolve_asof()
    log = {}
    df = load_df_official(ticker, asof, log=log)
    if df is None or len(df) < 30:
        raise ValueError(f"數據不足或 ticker 無效：{ticker}")
    if spy_df is None:
        spy_df = load_df_official("SPY", asof, log={})
    c = df["Close"]
    last = float(c.iloc[-1])
    has_vol = "Volume" in df.columns and float(df["Volume"].fillna(0).sum()) > 0
    vol20 = float((c * df["Volume"]).tail(20).mean()) if has_vol else np.nan
    atr50 = float(atr(df, 50).iloc[-1])
    atr5 = float(atr(df, 5).iloc[-1])
    atr50p = atr50 / last * 100
    atr5p = atr5 / last * 100
    ratio = atr5 / atr50 if atr50 else np.nan
    r3 = last / float(c.iloc[-64]) - 1 if len(c) > 64 else np.nan
    r1 = last / float(c.iloc[-22]) - 1 if len(c) > 22 else np.nan
    r6 = last / float(c.iloc[-127]) - 1 if len(c) > 127 else np.nan
    spy_r3 = float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-64]) - 1
    rel3_ppt = (r3 - spy_r3) * 100 if not np.isnan(r3) else np.nan
    # Always rank live returns vs universe (do not require ticker in rs.json)
    rs1 = rs_percentile(r1, "r1")
    rs3 = rs_percentile(r3, "r3")
    rs6 = rs_percentile(r6, "r6")
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    sma200 = c.rolling(200).mean()
    rsi = calc_rsi(c)
    macd, macds, macdh = calc_macd(c)
    k, d, j = calc_kdj(df)
    _e9, _e21, _e50 = float(ema9.iloc[-1]), float(ema21.iloc[-1]), float(ema50.iloc[-1])
    _mid = (_e9 + _e21 + _e50) / 3
    spread = (max(_e9, _e21, _e50) - min(_e9, _e21, _e50)) / last * 100
    dist = abs(last - _mid) / _mid * 100 if _mid else np.nan
    meta = universe_meta()
    return dict(
        spread=spread, dist=dist, above200=last >= float(sma200.iloc[-1]),
        df=df, last=last, vol20=vol20, has_vol=has_vol,
        atr50=atr50, atr5=atr5, atr50p=atr50p, atr5p=atr5p, ratio=ratio,
        rel3_ppt=rel3_ppt, r1=r1, r3=r3, r6=r6, rs1=rs1, rs3=rs3, rs6=rs6,
        ema9=ema9, ema21=ema21, ema50=ema50, sma200=sma200,
        rsi=rsi, macd=macd, macds=macds, macdh=macdh, k=k, d=d, j=j,
        asof=str(df.index[-1].date()),
        above50=last >= float(ema50.iloc[-1]),
        rs_universe_asof=meta["asof"],
        rs_universe_n=meta["n"],
        load_note=log.get(ticker, {}),
    )


def _draw_panels(fig, metrics, top=0.74, left=0.07, right=0.78, earn_label=""):
    df = metrics["df"].tail(130).copy()
    show_vol = metrics["has_vol"]
    if show_vol:
        gs = fig.add_gridspec(5, 1, height_ratios=[3.0, 0.85, 0.95, 0.95, 1.05], hspace=0.08,
                             left=left, right=right, top=top, bottom=0.06)
        axes = [fig.add_subplot(gs[i]) for i in range(5)]
        ax_p, ax_v, ax_kdj, ax_rsi, ax_macd = axes
    else:
        gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 1.0, 1.0, 1.1], hspace=0.08,
                             left=left, right=right, top=top, bottom=0.06)
        ax_p = fig.add_subplot(gs[0])
        ax_v = None
        ax_kdj = fig.add_subplot(gs[1], sharex=ax_p)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_p)
        ax_macd = fig.add_subplot(gs[3], sharex=ax_p)

    ema9 = metrics["ema9"].reindex(df.index)
    ema21 = metrics["ema21"].reindex(df.index)
    ema50 = metrics["ema50"].reindex(df.index)
    sma200 = metrics["sma200"].reindex(df.index)
    rsi = metrics["rsi"].reindex(df.index)
    macd = metrics["macd"].reindex(df.index)
    macds = metrics["macds"].reindex(df.index)
    macdh = metrics["macdh"].reindex(df.index)
    k = metrics["k"].reindex(df.index)
    d = metrics["d"].reindex(df.index)
    j = metrics["j"].reindex(df.index)
    x = np.arange(len(df))
    o, h, l, cl = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    up = cl >= o
    width = 0.65
    for i in range(len(df)):
        ax_p.vlines(x[i], l[i], h[i], color="#111111", linewidth=0.55, zorder=2)
        body_lo, body_hi = min(o[i], cl[i]), max(o[i], cl[i])
        fc = "white" if up[i] else "#111111"
        ax_p.add_patch(plt.Rectangle(
            (x[i] - width / 2, body_lo), width, max(body_hi - body_lo, 1e-9),
            facecolor=fc, edgecolor="#111111", linewidth=0.65, zorder=3))
    ax_p.plot(x, ema9.values, color="#7b2cbf", lw=1.15, label="EMA9", zorder=4)
    ax_p.plot(x, ema21.values, color="#2a9d4a", lw=1.15, label="EMA21", zorder=4)
    ax_p.plot(x, ema50.values, color="#d62828", lw=1.2, label="EMA50", zorder=4)
    if sma200.notna().sum() > 20:
        ax_p.plot(x, sma200.values, color="#1d3557", lw=1.25, label="SMA200", zorder=4)
    ax_p.legend(loc="upper left", fontsize=7, frameon=False, ncol=4)
    ax_p.grid(True, alpha=0.25, linestyle="--")
    ax_p.spines["top"].set_visible(False)
    ax_p.annotate(f"{float(cl[-1]):.2f}", xy=(x[-1], float(cl[-1])), xytext=(6, 0),
                  textcoords="offset points", fontsize=8, fontweight="bold", color="white", va="center",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#6c757d", ec="none"))
    dates = df.index
    tick_idx, last_m = [], None
    for i, dt in enumerate(dates):
        m = (dt.year, dt.month)
        if m != last_m:
            tick_idx.append(i)
            last_m = m
    if len(tick_idx) > 1 and tick_idx[1] - tick_idx[0] < 10:
        tick_idx = tick_idx[1:]
    if ax_v is not None:
        cols = ["#7ec8c8" if up[i] else "#f4a4a4" for i in range(len(df))]
        ax_v.bar(x, df["Volume"].values, color=cols, width=0.8)
        ax_v.set_ylabel("成交量", fontsize=7, color="#555")
        ax_v.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_vol_fmt))
        ax_v.grid(True, alpha=0.2, linestyle="--")
        ax_v.spines["top"].set_visible(False)
        plt.setp(ax_v.get_xticklabels(), visible=False)
    ax_kdj.plot(x, k.values, color="#e67e22", lw=1.0, label="K")
    ax_kdj.plot(x, d.values, color="#2980b9", lw=1.0, label="D")
    ax_kdj.plot(x, j.values, color="#8e44ad", lw=0.9, label="J", alpha=0.85)
    ax_kdj.axhline(80, color="#999", lw=0.6, ls="--")
    ax_kdj.axhline(20, color="#999", lw=0.6, ls="--")
    ax_kdj.set_ylabel("KDJ", fontsize=7)
    ax_kdj.legend(loc="upper left", fontsize=6, frameon=False, ncol=3)
    ax_kdj.grid(True, alpha=0.2, linestyle="--")
    ax_kdj.spines["top"].set_visible(False)
    plt.setp(ax_kdj.get_xticklabels(), visible=False)
    ax_rsi.plot(x, rsi.values, color="#16a085", lw=1.1, label="RSI14")
    ax_rsi.axhline(70, color="#c0392b", lw=0.6, ls="--")
    ax_rsi.axhline(30, color="#27ae60", lw=0.6, ls="--")
    ax_rsi.axhline(50, color="#999", lw=0.5, ls=":")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", fontsize=7)
    ax_rsi.legend(loc="upper left", fontsize=6, frameon=False)
    ax_rsi.grid(True, alpha=0.2, linestyle="--")
    ax_rsi.spines["top"].set_visible(False)
    annotate_rsi(ax_rsi, rsi)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in macdh.fillna(0).values]
    ax_macd.bar(x, macdh.fillna(0).values, color=colors, width=0.8, alpha=0.85)
    ax_macd.plot(x, macd.values, color="#2c3e50", lw=1.0, label="MACD")
    ax_macd.plot(x, macds.values, color="#e74c3c", lw=1.0, label="Signal")
    ax_macd.axhline(0, color="#666", lw=0.5)
    ax_macd.set_ylabel("MACD", fontsize=7)
    ax_macd.legend(loc="upper left", fontsize=6, frameon=False, ncol=2)
    ax_macd.grid(True, alpha=0.2, linestyle="--")
    ax_macd.spines["top"].set_visible(False)
    ax_macd.set_xticks(tick_idx)
    ax_macd.set_xticklabels([dates[i].strftime("%b") for i in tick_idx], fontsize=7)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    if earn_label:
        xi = tick_idx[-1] if tick_idx else len(df) - 1
        ax_macd.annotate(
            earn_label, xy=(xi, 0), xycoords=("data", "axes fraction"),
            xytext=(8, -16), textcoords="offset points",
            fontsize=6.5, color="#6a6a6a", ha="left", va="top", clip_on=False, zorder=6)
    return ax_p


def _pct_ret(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:+.1f}%"


def _rs_note(metrics) -> str:
    price_asof = metrics["asof"]
    uni_asof = metrics.get("rs_universe_asof") or ""
    n = metrics.get("rs_universe_n") or ""
    note = f"截至 {price_asof}；RS vs≈{n}隻（宇宙 asof {uni_asof}）"
    if uni_asof and price_asof != uni_asof:
        note += f"；價日≠宇宙日"
    return note


def render_index_card(ticker, title, metrics, note="") -> bytes:
    fig = plt.figure(figsize=(13.5, 8.2), dpi=150, facecolor="white")
    fig.text(0.05, 0.955, f"{ticker}  {title}", fontsize=18, fontweight="bold", color="#0d4f5b", va="top")
    ma = "高於 EMA50" if metrics["above50"] else "低於 EMA50"
    r1s = _pct_ret(metrics["r1"])
    r3s = _pct_ret(metrics["r3"])
    e9 = float(metrics["ema9"].iloc[-1])
    e21 = float(metrics["ema21"].iloc[-1])
    e50 = float(metrics["ema50"].iloc[-1])
    head2 = f"收市 {metrics['last']:.2f} ｜ {ma} ｜ EMA9 {e9:.2f} / EMA21 {e21:.2f} / EMA50 {e50:.2f} ｜ 1M {r1s} ｜ 3M {r3s}"
    fig.text(0.05, 0.915, head2, fontsize=9, color="#444", va="top")
    if note:
        fig.text(0.05, 0.885, note, fontsize=8, color="#666", va="top")
    ax_m = fig.add_axes([0.83, 0.12, 0.16, 0.68])
    ax_m.axis("off")
    rel = metrics["rel3_ppt"]
    rel_txt = (("跑贏" if rel >= 0 else "跑輸") + f" {abs(rel):.1f} ppt") if rel is not None and not np.isnan(rel) else "—"
    lines = [
        ("收市", f"{metrics['last']:.2f}"),
        ("成交額(20日)", fmt_vol(metrics["vol20"]) if metrics["has_vol"] else "—（無成交量）"),
        ("ATR50", f"${metrics['atr50']:.2f} / {metrics['atr50p']:.1f}%"),
        ("ATR5", f"${metrics['atr5']:.2f} / {metrics['atr5p']:.1f}% (x{metrics['ratio']:.2f})"),
        ("3M vs SPY", rel_txt),
        ("RSI", f"RSI={float(metrics['rsi'].iloc[-1]):.1f}"),
    ]
    y = 0.99
    for lab, val in lines:
        ax_m.text(0.0, y, lab, fontsize=7.2, color="#888", transform=ax_m.transAxes, va="top")
        ax_m.text(0.0, y - 0.028, val, fontsize=9.0, fontweight="bold", color="#111", transform=ax_m.transAxes, va="top")
        y -= 0.095
    ax_m.text(0.0, y, "RS", fontsize=9.5, fontweight="bold", color="#0d4f5b", transform=ax_m.transAxes, va="top")
    y -= 0.038
    for lab, rs, ret in [
        ("1個月", metrics.get("rs1"), metrics.get("r1")),
        ("3個月", metrics.get("rs3"), metrics.get("r3")),
        ("半年", metrics.get("rs6"), metrics.get("r6")),
    ]:
        ax_m.text(0.0, y, lab, fontsize=7.5, color="#555", transform=ax_m.transAxes, va="top")
        rs_s = "—" if rs is None else str(int(rs))
        ax_m.text(0.38, y, rs_s, fontsize=12.0, fontweight="bold", color=rs_color(rs), transform=ax_m.transAxes, va="top")
        ax_m.text(0.62, y, _pct_ret(ret), fontsize=7.5, color="#333", transform=ax_m.transAxes, va="top")
        y -= 0.048
    ax_m.text(0.0, 0.01, _rs_note(metrics), fontsize=5.5, color="#888", transform=ax_m.transAxes, va="bottom")
    _draw_panels(fig, metrics, top=0.78)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, facecolor="white", format="png")
    plt.close(fig)
    return buf.getvalue()


def render_stock_card(ticker, name, sector, industry, metrics,
                      blurb="", theme="", earn_label="", mcap_txt="—") -> bytes:
    fig = plt.figure(figsize=(13.5, 8.2), dpi=150, facecolor="white")
    fig.text(0.05, 0.990, f"{ticker}  {name}", fontsize=16, fontweight="bold", color="#0d4f5b", va="top")
    theme_txt = theme if theme else sector
    fig.text(0.05, 0.952, f"所屬版塊：{theme_txt}", fontsize=11, fontweight="bold", color="#0b6e4f", va="top")
    fig.text(0.05, 0.929, f"Sector: {sector}  |  Industry: {industry}", fontsize=8, color="#666", va="top")
    by = 0.909
    if blurb:
        for ln in wrap_chars(blurb, 78):
            fig.text(0.05, by, ln, fontsize=8, color="#222", va="top")
            by -= 0.015
    ax_m = fig.add_axes([0.795, 0.10, 0.19, 0.64])
    ax_m.axis("off")
    rel = metrics["rel3_ppt"]
    rel_txt = (("跑贏" if rel >= 0 else "跑輸") + f" {abs(rel):.1f} 個百分點") if rel is not None and not np.isnan(rel) else "—"
    lines = [
        ("收市 / 成交額", f"{metrics['last']:.2f} / {fmt_vol(metrics['vol20'])}", "#111"),
        ("ATR 50 日平均", f"${metrics['atr50']:.2f} / {metrics['atr50p']:.1f}%", "#111"),
        ("ATR 5 日平均", f"${metrics['atr5']:.2f} / {metrics['atr5p']:.1f}% (x{metrics['ratio']:.2f})", "#111"),
        ("3 個月 vs SPY", rel_txt, "#111"),
        ("均線匯合", f"spread {metrics['spread']:.2f}% / 距簇 {metrics['dist']:.2f}%", "#111"),
        ("市值", mcap_txt, "#111"),
    ]
    y = 0.99
    for lab, val, col in lines:
        ax_m.text(0.0, y, lab, fontsize=7.2, color="#888", transform=ax_m.transAxes, va="top")
        ax_m.text(0.0, y - 0.028, val, fontsize=9.0, fontweight="bold", color=col, transform=ax_m.transAxes, va="top")
        y -= 0.095
    ax_m.text(0.0, y, "RS", fontsize=9.5, fontweight="bold", color="#0d4f5b", transform=ax_m.transAxes, va="top")
    y -= 0.040
    for lab, rs, ret in [
        ("1個月", metrics.get("rs1"), metrics.get("r1")),
        ("3個月", metrics.get("rs3"), metrics.get("r3")),
        ("半年", metrics.get("rs6"), metrics.get("r6")),
    ]:
        ax_m.text(0.0, y, lab, fontsize=8.0, color="#555", transform=ax_m.transAxes, va="top")
        rs_s = "—" if rs is None else str(int(rs))
        ax_m.text(0.38, y, rs_s, fontsize=12.5, fontweight="bold", color=rs_color(rs), transform=ax_m.transAxes, va="top")
        ax_m.text(0.62, y, _pct_ret(ret), fontsize=8.5, color="#333", transform=ax_m.transAxes, va="top")
        y -= 0.055
    footnote = _rs_note(metrics) + "；21/63/126日回報百分位(1–99)。"
    fy = min(y - 0.01, 0.14)
    for ln in wrap_chars(footnote, 28):
        ax_m.text(0.0, fy, ln, fontsize=5.6, color="#888", transform=ax_m.transAxes, va="top")
        fy -= 0.032
    _draw_panels(fig, metrics, top=0.74, earn_label=earn_label)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, facecolor="white", format="png")
    plt.close(fig)
    return buf.getvalue()


def build_card(ticker: str, asof: Optional[str] = None) -> dict:
    """Build card PNG + text summary for a ticker. Raises ValueError on bad ticker."""
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 16:
        raise ValueError("無效 ticker")
    asof = asof or resolve_asof()
    info = _ticker_info(ticker)
    # Validate ticker exists
    if not info and ticker not in KNOWN_INDEX:
        # try a quick download probe
        try:
            probe = load_df_official(ticker, asof, log={})
            if probe is None or len(probe) < 10:
                raise ValueError(f"搵唔到 {ticker}")
        except Exception as e:
            raise ValueError(f"搵唔到 {ticker}：{e}") from e
    kind = detect_kind(ticker, info)
    metrics = compute_metrics(ticker, asof=asof)
    name = info.get("shortName") or info.get("longName") or KNOWN_INDEX.get(ticker, ticker)
    sector = info.get("sector") or "—"
    industry = info.get("industry") or "—"
    mcap = info.get("marketCap")
    mcap_txt = fmt_mcap(mcap)
    earn = "" if kind == "index" else next_earnings_label(ticker, info)
    if kind == "index":
        title = KNOWN_INDEX.get(ticker) or name
        note = f"指數／ETF 卡｜數據 Yahoo 官方收市（未經股息調整）"
        png = render_index_card(ticker, title, metrics, note=note)
        display_name = title
    else:
        blurb = f"市值：{mcap_txt}。" if mcap_txt != "—" else ""
        png = render_stock_card(
            ticker, name, sector, industry, metrics,
            blurb=blurb, theme=sector, earn_label=earn, mcap_txt=mcap_txt,
        )
        display_name = name
    summary = {
        "ticker": ticker,
        "name": display_name,
        "kind": kind,
        "sector": sector,
        "industry": industry,
        "last": metrics["last"],
        "asof": metrics["asof"],
        "rs1": metrics["rs1"],
        "rs3": metrics["rs3"],
        "rs6": metrics["rs6"],
        "r1": None if metrics["r1"] is None or (isinstance(metrics["r1"], float) and np.isnan(metrics["r1"])) else float(metrics["r1"]),
        "r3": None if metrics["r3"] is None or (isinstance(metrics["r3"], float) and np.isnan(metrics["r3"])) else float(metrics["r3"]),
        "r6": None if metrics["r6"] is None or (isinstance(metrics["r6"], float) and np.isnan(metrics["r6"])) else float(metrics["r6"]),
        "atr50": metrics["atr50"],
        "atr5": metrics["atr5"],
        "rel3_ppt": None if np.isnan(metrics["rel3_ppt"]) else float(metrics["rel3_ppt"]),
        "rsi": float(metrics["rsi"].iloc[-1]),
        "mcap": mcap_txt,
        "earn_label": earn,
        "rs_universe_asof": metrics["rs_universe_asof"],
        "rs_universe_n": metrics["rs_universe_n"],
        "rs_note": _rs_note(metrics),
    }
    return {"png": png, "summary": summary}
