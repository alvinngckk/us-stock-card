# 美股查股卡（自架網站）

輸入 ticker，輸出同每日 PDF **個股／指數頁**一樣風格嘅圖卡（K線＋EMA／成交量／KDJ／RSI／MACD＋右邊指標同 **RS**）。

## 點開（Alvin）

伺服器已綁 `0.0.0.0:8765`。

1. **本機／盒內瀏覽器**（最穩）：
   - http://127.0.0.1:8765
   - http://127.0.0.1:8765/?t=AAPL
   - http://127.0.0.1:8765/stock/JNJ
2. **喺 Grok Bot 開 computer preview**：揀「選股」agent 名稱 → 開電腦預覽 → 瀏覽器入上述網址。
3. 公網隧道（cloudflared／localtunnel／bore）喺呢個環境出站 7844 等埠被擋，暫無穩定 https 公網 URL。

範例按鈕：AAPL、NVDA、JNJ、SPY、QQQ、TLT、IWM。

## 啟動／重啟

```bash
cd /workspace/stockpick/webapp
./start.sh
# 日誌 server.log ； PID server.pid
curl http://127.0.0.1:8765/health
```

## RS 說明

- **1個月／3個月／半年** = 21／63／126 個交易日未經股息調整收市回報
- 百分位 **1–99**，對齊 `d1009/rs.json` 股票宇宙（約 5,537 隻）
- 新 ticker **唔使**已喺 rs.json：用即時回報對宇宙分佈排名
- ≥70 深綠 `#006b45`；≤30 深紅 `#8b0000`
- 頁面標 RS 宇宙 asof；若價日同宇宙日唔同會註明

## 注意

- 價格經 Yahoo／`dataio.load_df_official`（可有延遲）
- 首次查約 5–15 秒；之後 5 分鐘快取
- 樣本圖：`sample_AAPL.png`、`sample_SPY.png`
- 僅供研究，唔構成投資建議
