import pandas as pd, numpy as np, yfinance as yf

def _flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def patch_last(df, ticker):
    """Yahoo sometimes returns NaN OHLC for the latest session; rebuild from 30m bars."""
    if len(df) == 0 or not pd.isna(df["Close"].iloc[-1]):
        return df
    last_day = df.index[-1].date()
    try:
        ib = _flat(yf.download(ticker, period="5d", interval="30m", progress=False, auto_adjust=True, threads=False))
        ib = ib[[d.date() == last_day for d in ib.index.tz_convert("America/New_York")]] if ib.index.tz is not None else ib[[d.date()==last_day for d in ib.index]]
        if len(ib):
            df.loc[df.index[-1], "Open"] = float(ib["Open"].iloc[0])
            df.loc[df.index[-1], "High"] = float(ib["High"].max())
            df.loc[df.index[-1], "Low"] = float(ib["Low"].min())
            df.loc[df.index[-1], "Close"] = float(ib["Close"].iloc[-1])
            if pd.isna(df["Volume"].iloc[-1]) or df["Volume"].iloc[-1] == 0:
                df.loc[df.index[-1], "Volume"] = float(ib["Volume"].sum())
    except Exception as e:
        print("patch fail", ticker, e)
    return df

def load_df(ticker, period="2y"):
    df = _flat(yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False))
    df = df.dropna(how="all")
    df = patch_last(df, ticker)
    return df.dropna(subset=["Close"])


# ---- v7 (2026-10-07): official-close loader ----
_INFO_CACHE = {}
def _info(ticker):
    if ticker not in _INFO_CACHE:
        try:
            _INFO_CACHE[ticker] = yf.Ticker(ticker).info
        except Exception as e:
            print("info fail", ticker, e); _INFO_CACHE[ticker] = {}
    return _INFO_CACHE[ticker]

def load_df_official(ticker, asof, period="2y", log=None, auto_adjust=False):
    """Daily OHLCV truncated at `asof` (YYYY-MM-DD, ET session date).
    - equities/ETFs: last bar rebuilt from 30m bars if NaN, then OHLCV overwritten by Yahoo
      quote fields (regularMarketPrice/Open/DayHigh/DayLow/Volume) when that quote is for `asof`.
    - DX-Y.NYB / 24h instruments: last bar rebuilt from 30m bars (prev 18:00 ET -> asof 17:00 ET),
      close = Yahoo regularMarketPreviousClose when the live quote has already rolled to the next session.
    """
    asof = pd.Timestamp(asof)
    # v10c (2026-10-08): default UNADJUSTED (split-adjusted only) closes so SMA/EMA/52w-high/ATR match Finviz/Nasdaq/StockCharts;
    # the official-quote overwrite of the last bar is unadjusted too, so history and last bar are now consistent.
    df = _flat(yf.download(ticker, period=period, auto_adjust=auto_adjust, progress=False, threads=False))
    if "Adj Close" in df.columns: df = df.drop(columns=["Adj Close"])
    df = df.dropna(how="all")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else df.index
    df = df.loc[:asof].copy()
    info = _info(ticker)
    qt = info.get("quoteType", "")
    rmt = info.get("regularMarketTime")
    rmt_d = pd.Timestamp(rmt, unit="s", tz="UTC").tz_convert("America/New_York").date() if rmt else None
    note = []
    if asof not in df.index:
        df.loc[asof] = np.nan
    if ticker == "DX-Y.NYB" or qt in ("CURRENCY", "FUTURE"):
        ib = _flat(yf.download(ticker, period="5d", interval="30m", progress=False, auto_adjust=auto_adjust, threads=False))
        ib.index = ib.index.tz_convert("America/New_York")
        lo = pd.Timestamp(asof.date() - pd.Timedelta(days=1)).tz_localize("America/New_York") + pd.Timedelta(hours=18)
        hi = pd.Timestamp(asof.date()).tz_localize("America/New_York") + pd.Timedelta(hours=17)
        w = ib[(ib.index >= lo) & (ib.index < hi)]
        if len(w):
            close = float(w["Close"].iloc[-1])
            if rmt_d and rmt_d > asof.date() and info.get("regularMarketPreviousClose"):
                close = float(info["regularMarketPreviousClose"]); note.append("close=prevClose")
            df.loc[asof, ["Open", "High", "Low", "Close"]] = [float(w["Open"].iloc[0]), float(w["High"].max()), float(w["Low"].min()), close]
            df.loc[asof, "Volume"] = 0.0
            note.append(f"rebuilt from {len(w)}x30m")
    else:
        if pd.isna(df.loc[asof, "Close"]):
            ib = _flat(yf.download(ticker, period="5d", interval="30m", progress=False, auto_adjust=auto_adjust, threads=False))
            ib.index = ib.index.tz_convert("America/New_York")
            w = ib[[d.date() == asof.date() and d.hour < 16 for d in ib.index]]
            if len(w):
                df.loc[asof, ["Open", "High", "Low", "Close", "Volume"]] = [float(w["Open"].iloc[0]), float(w["High"].max()), float(w["Low"].min()), float(w["Close"].iloc[-1]), float(w["Volume"].sum())]
                note.append("patched 30m")
        if rmt_d == asof.date() and info.get("regularMarketPrice"):
            df.loc[asof, "Close"] = float(info["regularMarketPrice"])
            for col, k in (("Open", "regularMarketOpen"), ("High", "regularMarketDayHigh"), ("Low", "regularMarketDayLow"), ("Volume", "regularMarketVolume")):
                if info.get(k): df.loc[asof, col] = float(info[k])
            note.append("official quote")
    df = df.dropna(subset=["Close"])
    if log is not None: log[ticker] = dict(last=str(df.index[-1].date()), close=float(df["Close"].iloc[-1]), note=";".join(note))
    return df
