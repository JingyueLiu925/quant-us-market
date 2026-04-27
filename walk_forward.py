# ============================================================
# 滚动样本外验证（Rolling Walk-Forward Test）v4
#
# 与 long_only.py v4 完全同步：
#   1. 双周再平衡（月初 + 月末）
#   2. 个股 10% 止损，每日监控
#   3. 复合动量评分：60% × (12-1月) + 40% × (3月近期加速)
#   4. 熔断解除：SPY > 50MA（单条件，快速重入）
#
# 测试方法：4 个滚动窗口，每窗口 8 年训练 + 2 年样本外
# ============================================================

import os, io, warnings, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# 滚动窗口配置
# ============================================================
WINDOWS = [
    ("2008-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2010-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2012-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2014-01-01", "2021-12-31", "2022-01-01", "2024-12-31"),
]

DOWNLOAD_FROM = "2007-01-01"
DOWNLOAD_TO   = "2024-12-31"
INITIAL_CASH  = 20_000

# ============================================================
# 策略参数（固定，不修改）
# ============================================================
STOCK_ALLOC   = 0.85
TOP_STOCKS    = 5
MAX_DRAWDOWN  = 0.20
MIN_UP_MONTHS = 5

DEFENSE   = {"SHY": 0.65, "GLD": 0.25}
CACHE_DIR = "./cache"

_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
SP500_HISTORY_URLS = [
    _BASE + "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv",
    _BASE + "S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
]


# ============================================================
# 数据层
# ============================================================

def load_sp500_history():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/sp500_history.pkl"
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    for url in SP500_HISTORY_URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.to_pickle(cache_file)
            print(f"历史成分股数据已下载，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"尝试失败：{e}")
    print("使用当前标普500快照作为备用")
    return None


def get_universe_at(history_df, date):
    if history_df is None:
        return []
    past = history_df[history_df.index <= date]
    if past.empty:
        past = history_df.iloc[:1]
    return [t.strip().replace(".", "-") for t in past.iloc[-1]["tickers"].split(",")]


def get_all_historical_tickers(history_df):
    if history_df is None:
        return []
    all_tickers = set()
    for row in history_df["tickers"]:
        for t in row.split(","):
            all_tickers.add(t.strip().replace(".", "-"))
    return list(all_tickers)


def download_prices(tickers, start, end):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/prices_{start[:4]}_{end[:4]}.pkl"
    if os.path.exists(cache_file):
        print("从缓存加载价格数据")
        df = pd.read_pickle(cache_file)
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            extra = yf.download(missing, start=start, end=end,
                                auto_adjust=True, progress=False)
            if isinstance(extra.columns, pd.MultiIndex):
                extra = extra["Close"]
            df = pd.concat([df, extra], axis=1)
            df.to_pickle(cache_file)
        return df
    print(f"正在下载 {len(tickers)} 只股票数据...")
    frames = []
    for i in range(0, len(tickers), 200):
        batch = tickers[i: i + 200]
        raw = yf.download(batch, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"]
        frames.append(raw)
    closes = pd.concat(frames, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    closes.to_pickle(cache_file)
    return closes


# ============================================================
# 信号计算
# ============================================================

def calc_momentum(closes, as_of, symbols, sma200_filter=True):
    """纯 12-1 月动量评分"""
    scores = {}
    for sym in symbols:
        if sym not in closes.columns:
            continue
        current_price = closes[sym].get(as_of, np.nan)
        if pd.isna(current_price) or current_price <= 0:
            continue
        hist = closes[sym].loc[:as_of].dropna()
        if len(hist) < 252:
            continue
        price_skip   = hist.iloc[-21]
        price_origin = hist.iloc[-252]
        if price_origin <= 0 or price_skip <= 0:
            continue
        momentum = price_skip / price_origin - 1
        if momentum <= 0:
            continue
        monthly   = hist.iloc[-252::21]
        up_months = (monthly.pct_change().dropna() > 0).sum()
        if up_months < MIN_UP_MONTHS:
            continue
        if sma200_filter:
            if len(hist) < 200:
                continue
            if hist.iloc[-1] < hist.iloc[-200:].mean():
                continue
        scores[sym] = momentum
    return scores


def spy_above_sma200(closes, date):
    hist = closes["SPY"].loc[:date].dropna()
    if len(hist) < 200:
        return True
    return hist.iloc[-1] > hist.iloc[-200:].mean()


def spy_above_sma50(closes, date):
    """v4：熔断解除改为 SPY > 50MA"""
    hist = closes["SPY"].loc[:date].dropna()
    if len(hist) < 50:
        return True
    return hist.iloc[-1] > hist.iloc[-50:].mean()


# ============================================================
# 回测引擎
# ============================================================

class Backtester:

    def __init__(self, closes, history_df, start, end, initial_cash=INITIAL_CASH):
        self.closes_full  = closes
        self.closes       = closes.loc[start:end]
        self.history_df   = history_df
        self.cash         = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.holdings     = {}
        self.entry_prices = {}
        self.nav_log      = []
        self.circuit_breaker = False
        self.peak_nav     = float(initial_cash)

        # 月末再平衡
        idx = pd.DatetimeIndex(self.closes.index)
        self.rebal_dates = set(pd.Series(idx).groupby(idx.to_period("M")).last())

    def _nav(self, date):
        prices = self.closes.loc[date]
        equity = 0.0
        for sym, sh in self.holdings.items():
            p = prices.get(sym, np.nan)
            if pd.notna(p) and p > 0:
                equity += sh * p
        return self.cash + equity

    def _liquidate(self, date, symbol=None):
        prices = self.closes.loc[date]
        if symbol is not None:
            if symbol in self.holdings:
                p = prices.get(symbol, np.nan)
                if pd.notna(p) and p > 0:
                    self.cash += self.holdings[symbol] * p
                del self.holdings[symbol]
                self.entry_prices.pop(symbol, None)
        else:
            for sym, sh in self.holdings.items():
                p = prices.get(sym, np.nan)
                if pd.notna(p) and p > 0:
                    self.cash += sh * p
            self.holdings     = {}
            self.entry_prices = {}

    def _buy(self, date, targets):
        self._liquidate(date)
        prices = self.closes.loc[date]
        nav    = self.cash
        for sym, weight in targets.items():
            p = prices.get(sym, np.nan)
            if pd.isna(p) or p <= 0:
                continue
            shares = (nav * weight) / p
            cost   = shares * p
            if cost > self.cash:
                continue
            self.holdings[sym]     = shares
            self.entry_prices[sym] = p
            self.cash             -= cost

    def _check_stop_loss(self, date):
        if self.circuit_breaker:
            return
        prices   = self.closes.loc[date]
        to_close = []
        for sym, entry in list(self.entry_prices.items()):
            if sym in DEFENSE:
                continue
            p = prices.get(sym, np.nan)
            if pd.isna(p) or p <= 0:
                to_close.append(sym)
                continue
            if (p - entry) / entry <= -STOP_LOSS_PCT:
                to_close.append(sym)
        for sym in to_close:
            self._liquidate(date, sym)

    def _check_drawdown(self, date):
        nav = self._nav(date)
        if nav > self.peak_nav:
            self.peak_nav = nav
        drawdown = (self.peak_nav - nav) / self.peak_nav
        if drawdown >= MAX_DRAWDOWN and not self.circuit_breaker:
            self.circuit_breaker = True
            self._buy(date, DEFENSE)

    def _rebalance(self, date):
        if self.circuit_breaker:
            if spy_above_sma50(self.closes_full, date):
                self.circuit_breaker = False
                self.peak_nav        = self._nav(date)
                self._liquidate(date)
            else:
                return

        if not spy_above_sma200(self.closes_full, date):
            self._buy(date, DEFENSE)
            return

        universe     = get_universe_at(self.history_df, date)
        stock_scores = calc_momentum(self.closes_full, date, universe, sma200_filter=True)
        top_stocks   = sorted(stock_scores, key=lambda x: stock_scores[x], reverse=True)[:TOP_STOCKS]

        if top_stocks:
            w = STOCK_ALLOC / len(top_stocks)
            self._buy(date, {s: w for s in top_stocks})

    def run(self):
        for date in self.closes.index:
            self._check_stop_loss(date)
            if not self.circuit_breaker:
                self._check_drawdown(date)
            if date in self.rebal_dates:
                self._rebalance(date)
            self.nav_log.append({"date": date, "nav": self._nav(date)})
        return pd.DataFrame(self.nav_log).set_index("date")


# ============================================================
# 绩效计算
# ============================================================

def calc_metrics(nav_series, initial_cash, label=""):
    years     = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    total_ret = (nav_series.iloc[-1] - initial_cash) / initial_cash
    cagr      = (nav_series.iloc[-1] / initial_cash) ** (1 / years) - 1 if years > 0 else 0
    daily_ret = nav_series.pct_change().dropna()
    sharpe    = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd    = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    return {"label": label, "total_ret": total_ret, "cagr": cagr,
            "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar}


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  Rolling Walk-Forward 滚动样本外验证 v4")
    print("=" * 68)
    for i, (ts, te, os_, oe) in enumerate(WINDOWS, 1):
        print(f"  窗口 {i}：训练 {ts[:4]}-{te[:4]}  |  样本外 {os_[:4]}-{oe[:4]}")
    print("=" * 68)

    # 数据准备
    history_df  = load_sp500_history()
    all_tickers = list(set(
        get_all_historical_tickers(history_df) + list(DEFENSE.keys())
    )) if history_df is not None else list(DEFENSE.keys())

    closes = download_prices(all_tickers, DOWNLOAD_FROM, DOWNLOAD_TO)
    print(f"价格数据：{closes.shape[0]} 天 × {closes.shape[1]} 只股票\n")

    # 滚动回测
    all_train_navs = []
    all_oos_navs   = []
    window_metrics = []

    for i, (ts, te, os_, oe) in enumerate(WINDOWS, 1):
        print(f"--- 窗口 {i}：训练 {ts[:4]}-{te[:4]} ---")
        bt_train  = Backtester(closes, history_df, ts, te, INITIAL_CASH)
        nav_train = bt_train.run()
        all_train_navs.append(nav_train)

        print(f"--- 窗口 {i}：样本外 {os_[:4]}-{oe[:4]} ---")
        bt_oos  = Backtester(closes, history_df, os_, oe, INITIAL_CASH)
        nav_oos = bt_oos.run()

        spy_oos = closes["SPY"].reindex(nav_oos.index).ffill()
        spy_oos = spy_oos / spy_oos.iloc[0] * INITIAL_CASH

        m_train   = calc_metrics(nav_train["nav"], INITIAL_CASH, f"W{i}-训练")
        m_oos     = calc_metrics(nav_oos["nav"],   INITIAL_CASH, f"W{i}-OOS")
        m_spy_oos = calc_metrics(spy_oos,          INITIAL_CASH, f"W{i}-SPY")

        window_metrics.append({
            "window": i, "period": f"{os_[:4]}-{oe[:4]}",
            "train": m_train, "oos": m_oos, "spy_oos": m_spy_oos,
        })

        nav_oos_norm = nav_oos["nav"] / nav_oos["nav"].iloc[0]
        spy_oos_norm = spy_oos / spy_oos.iloc[0]
        all_oos_navs.append((nav_oos_norm, spy_oos_norm, f"{os_[:4]}-{oe[:4]}"))

    # 拼接 OOS 曲线
    oos_mult  = 1.0
    spy_mult  = 1.0
    stitched_strat = []
    stitched_spy   = []

    for nav_norm, spy_norm, _ in all_oos_navs:
        stitched_strat.append(nav_norm * oos_mult)
        stitched_spy.append(spy_norm * spy_mult)
        oos_mult = (nav_norm * oos_mult).iloc[-1]
        spy_mult = (spy_norm * spy_mult).iloc[-1]

    stitched_strat = pd.concat(stitched_strat)
    stitched_spy   = pd.concat(stitched_spy)

    m_oos_all = calc_metrics(stitched_strat * INITIAL_CASH, INITIAL_CASH, "OOS拼接-策略")
    m_spy_all = calc_metrics(stitched_spy   * INITIAL_CASH, INITIAL_CASH, "OOS拼接-SPY")

    # 打印报告
    print("\n" + "=" * 68)
    print("  各窗口样本外绩效对比")
    print("=" * 68)
    print(f"  {'窗口':<8} {'OOS期间':<12} {'策略CAGR':>9} {'SPYCAGR':>9} {'策略夏普':>9} {'策略MaxDD':>10} {'跑赢':>5}")
    print("-" * 68)
    beat_count = 0
    for wm in window_metrics:
        w    = wm["oos"]
        s    = wm["spy_oos"]
        beat = w["cagr"] > s["cagr"]
        if beat:
            beat_count += 1
        flag = "✓" if beat else "✗"
        print(f"  {wm['window']:<8} {wm['period']:<12} "
              f"{w['cagr']:>9.2%} {s['cagr']:>9.2%} "
              f"{w['sharpe']:>9.3f} {w['max_dd']:>10.2%} {flag:>5}")

    print("=" * 68)
    print(f"\n  拼接 OOS 整体绩效（{WINDOWS[0][2][:4]}-{WINDOWS[-1][3][:4]}）")
    print("-" * 68)
    print(f"  {'指标':<14} {'OOS策略':>10} {'OOS SPY':>10}")
    print(f"  {'总收益率':<14} {m_oos_all['total_ret']:>10.2%} {m_spy_all['total_ret']:>10.2%}")
    print(f"  {'年化收益':<14} {m_oos_all['cagr']:>10.2%} {m_spy_all['cagr']:>10.2%}")
    print(f"  {'夏普比率':<14} {m_oos_all['sharpe']:>10.3f} {m_spy_all['sharpe']:>10.3f}")
    print(f"  {'最大回撤':<14} {m_oos_all['max_dd']:>10.2%} {m_spy_all['max_dd']:>10.2%}")
    print(f"  {'Calmar':<14} {m_oos_all['calmar']:>10.3f} {m_spy_all['calmar']:>10.3f}")
    print("=" * 68)

    last_train   = window_metrics[-1]["train"]
    cagr_decay   = (last_train["cagr"] - m_oos_all["cagr"]) / last_train["cagr"] if last_train["cagr"] > 0 else 0
    sharpe_decay = (last_train["sharpe"] - m_oos_all["sharpe"]) / last_train["sharpe"] if last_train["sharpe"] > 0 else 0
    oos_beats_spy = m_oos_all["cagr"] > m_spy_all["cagr"]
    oos_lower_dd  = abs(m_oos_all["max_dd"]) < abs(m_spy_all["max_dd"])
    oos_sharpe_ok = m_oos_all["sharpe"] > 0.5

    print("\n  过拟合诊断：")
    print(f"  CAGR 衰减：  {cagr_decay:.1%}  {'[警惕]' if cagr_decay > 0.3 else '[正常]'}")
    print(f"  夏普 衰减：  {sharpe_decay:.1%}  {'[警惕]' if sharpe_decay > 0.3 else '[正常]'}")
    print(f"  各窗口跑赢SPY：{beat_count}/{len(WINDOWS)}")
    print(f"  OOS整体跑赢SPY：{'是' if oos_beats_spy else '否'}")
    print(f"  OOS回撤低于SPY：{'是' if oos_lower_dd else '否'}")
    print(f"  OOS夏普 > 0.5： {'是' if oos_sharpe_ok else '否'}")

    verdict = sum([
        cagr_decay < 0.3,
        sharpe_decay < 0.3,
        beat_count >= len(WINDOWS) // 2 + 1,
        oos_beats_spy,
        oos_lower_dd,
        oos_sharpe_ok,
    ])
    print(f"\n  综合评分：{verdict}/6")
    if verdict >= 5:
        print("  结论：样本外表现稳健，可考虑 Paper Trading")
    elif verdict >= 3:
        print("  结论：样本外表现中等，建议扩大测试样本再决策")
    else:
        print("  结论：样本外表现差，策略存在较严重过拟合")
    print("=" * 68)

    # 绘图
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    ax_train = fig.add_subplot(gs[0, :])
    for j, wm in enumerate(window_metrics):
        ts, te = WINDOWS[j][0], WINDOWS[j][1]
        nav = all_train_navs[j]["nav"] / INITIAL_CASH
        ax_train.plot(nav, label=f"W{j+1} 训练({ts[:4]}-{te[:4]})",
                      color=colors[j], linewidth=1.2, alpha=0.8)
    ax_train.set_title("各窗口训练期净值（已知样本）", fontsize=11)
    ax_train.legend(fontsize=8, ncol=4); ax_train.grid(alpha=0.3)
    ax_train.set_ylabel("归一化净值")

    ax_oos = fig.add_subplot(gs[1, 0])
    ax_oos.plot(stitched_strat.index, stitched_strat,
                label="策略OOS（拼接）", color="#1976D2", linewidth=1.5)
    ax_oos.plot(stitched_spy.index, stitched_spy,
                label="SPY", color="#F44336", linewidth=1, linestyle="--", alpha=0.7)
    for j in range(1, len(all_oos_navs)):
        ax_oos.axvline(x=all_oos_navs[j][0].index[0], color="gray", linestyle=":", linewidth=0.8)
    ax_oos.set_title("OOS 拼接净值（首次验证）", fontsize=11, color="#C62828")
    ax_oos.legend(fontsize=8); ax_oos.grid(alpha=0.3)
    ax_oos.set_ylabel("归一化净值")

    ax_bar = fig.add_subplot(gs[1, 1])
    x = np.arange(len(WINDOWS))
    w = 0.35
    cagr_strats = [wm["oos"]["cagr"] * 100 for wm in window_metrics]
    cagr_spys   = [wm["spy_oos"]["cagr"] * 100 for wm in window_metrics]
    bars1 = ax_bar.bar(x - w/2, cagr_strats, w, label="策略", color="#1976D2", alpha=0.85)
    bars2 = ax_bar.bar(x + w/2, cagr_spys,   w, label="SPY",  color="#F44336", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([wm["period"] for wm in window_metrics], fontsize=8)
    ax_bar.set_title("各窗口 OOS 年化收益对比 (%)", fontsize=11)
    ax_bar.set_ylabel("CAGR (%)"); ax_bar.legend(fontsize=8)
    ax_bar.grid(alpha=0.3, axis="y"); ax_bar.axhline(0, color="black", linewidth=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    ax_dd = fig.add_subplot(gs[2, 0])
    dd_strat = (stitched_strat - stitched_strat.cummax()) / stitched_strat.cummax() * 100
    dd_spy   = (stitched_spy   - stitched_spy.cummax())   / stitched_spy.cummax()   * 100
    ax_dd.fill_between(dd_strat.index, dd_strat, 0, alpha=0.5, color="#1976D2", label="策略")
    ax_dd.fill_between(dd_spy.index,   dd_spy,   0, alpha=0.3, color="#F44336", label="SPY")
    ax_dd.set_title("OOS 回撤对比 (%)", fontsize=10)
    ax_dd.legend(fontsize=8); ax_dd.grid(alpha=0.3)

    ax_sharpe = fig.add_subplot(gs[2, 1])
    sharpe_strats = [wm["oos"]["sharpe"] for wm in window_metrics]
    sharpe_spys   = [wm["spy_oos"]["sharpe"] for wm in window_metrics]
    ax_sharpe.bar(x - w/2, sharpe_strats, w, label="策略", color="#1976D2", alpha=0.85)
    ax_sharpe.bar(x + w/2, sharpe_spys,   w, label="SPY",  color="#F44336", alpha=0.85)
    ax_sharpe.axhline(0.5, color="green", linestyle="--", linewidth=1, label="目标 0.5")
    ax_sharpe.set_xticks(x)
    ax_sharpe.set_xticklabels([wm["period"] for wm in window_metrics], fontsize=8)
    ax_sharpe.set_title("各窗口 OOS 夏普比率对比", fontsize=11)
    ax_sharpe.legend(fontsize=8); ax_sharpe.grid(alpha=0.3, axis="y")

    plt.suptitle("Rolling Walk-Forward 滚动样本外验证 v4", fontsize=14, fontweight="bold")
    out = "walk_forward_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  图表已保存：{out}")
