# ============================================================
# 长线动量策略 v4 — 本地回测版
# 依赖安装：pip install yfinance pandas numpy matplotlib requests
#
# 与 long_only.py v4 完全同步：
#   1. 双周再平衡（月初 + 月末）
#   2. 个股 10% 止损，每日监控
#   3. 复合动量评分：60% × (12-1月) + 40% × (3月近期加速)
#   4. 熔断解除：SPY > 50MA（单条件，快速重入）
#   5. 流动性门槛说明：本地使用历史标普500成分股（约500只），
#      已隐含大市值/高流动性过滤，与 QC 的 8000万门槛等效
# ============================================================

import os
import io
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import yfinance as yf

# macOS 中文字体：直接按文件路径加载，绕过字体缓存问题
for _fp in [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]:
    try:
        _fm.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from fundamental_data import load_simfin_data, build_fundamental_cache, get_fundamental_scores

warnings.filterwarnings("ignore")

# ============================================================
# 配置参数（与 long_only.py v4 对齐）
# ============================================================
START_DATE    = "2008-01-01"
END_DATE      = "2025-12-31"
INITIAL_CASH  = 20_000

STOCK_ALLOC   = 0.85
TOP_STOCKS    = 12     # 持仓从5只扩至12只，降低集中度

MAX_DRAWDOWN   = 0.20
MIN_UP_MONTHS  = 5
MAX_ANNUAL_VOL = 0.80

# 基本面过滤阈值（点位正确，使用 SimFin 历史季报）
MIN_ROE          =  0.10   # ROE > 10%（盈利能力）
MIN_GROSS_MARGIN =  0.25   # 毛利率 > 25%（商业模式护城河）
MIN_REV_GROWTH   = -0.05   # 营收同比 > -5%（允许轻微下滑，排除严重萎缩）
MIN_PIOTROSKI    =  5      # Piotroski F-Score >= 5（综合质量门槛，满分9）
DEFENSE   = {"SHY": 0.65, "GLD": 0.25}

# 长桥平台交易成本（单边）
# 佣金：免佣；SEC费：0.00278%；FINRA TAF：约0.002%；滑点：约0.05%
# 合计单边约 0.06%，买卖双边约 0.12%
TRADE_COST = 0.0006  # 单边成本率
CACHE_DIR = "./cache"

_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
SP500_HISTORY_URLS = [
    _BASE + "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv",
    _BASE + "S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
]


# ============================================================
# 1. 历史 Universe
# ============================================================

def load_sp500_history():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/sp500_history.pkl"

    if os.path.exists(cache_file):
        print("从缓存加载历史成分股数据")
        return pd.read_pickle(cache_file)

    print("正在下载历史标普500成分股数据...")
    for url in SP500_HISTORY_URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.to_pickle(cache_file)
            print(f"历史成分股数据已缓存，共 {len(df)} 条变动记录")
            return df
        except Exception as e:
            print(f"尝试失败（{e}），换下一个地址...")

    print("所有地址均失败，使用当前标普500快照作为备用")
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


# ============================================================
# 2. 价格数据下载
# ============================================================

def download_prices(tickers, start, end):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/prices_{start[:4]}_{end[:4]}.pkl"

    if os.path.exists(cache_file):
        print(f"从缓存加载价格数据：{cache_file}")
        df = pd.read_pickle(cache_file)
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            print(f"补充下载 {len(missing)} 只缓存中缺失的股票...")
            extra = yf.download(missing, start=start, end=end,
                                auto_adjust=True, progress=False)
            if isinstance(extra.columns, pd.MultiIndex):
                extra = extra["Close"]
            df = pd.concat([df, extra], axis=1)
            df.to_pickle(cache_file)
        return df

    print(f"正在下载 {len(tickers)} 只股票历史价格（首次较慢）...")
    frames = []
    for i in range(0, len(tickers), 200):
        batch = tickers[i: i + 200]
        print(f"  下载第 {i//200 + 1} 批（{len(batch)} 只）...")
        raw = yf.download(batch, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"]
        frames.append(raw)

    closes = pd.concat(frames, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    closes.to_pickle(cache_file)
    print(f"价格数据已缓存至：{cache_file}")
    return closes


# ============================================================
# 3. 信号计算
# ============================================================

def calc_momentum(closes, as_of, symbols, sma200_filter=True,
                  min_up_months=MIN_UP_MONTHS, vol_adjust=False):
    """
    12-1 月动量评分，叠加三重质量过滤：
    1. 动量质量：12个月中至少 min_up_months 个月上涨（趋势持续性）
    2. 波动率过滤：年化波动率 > MAX_ANNUAL_VOL 的股票排除
       例外：近60日成交额排名前10%的大盘股免除此限制（NVDA类强势大盘股）
    3. 200MA 过滤：个股价格须在 200MA 以上（仅 sma200_filter=True）
    """
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

        # 过滤1：动量质量（趋势持续性）
        monthly   = hist.iloc[-252::21]
        up_months = (monthly.pct_change().dropna() > 0).sum()
        if up_months < min_up_months:
            continue

        # 过滤2：波动率过滤（统一上限80%，允许NVDA类高波动强势股入场）
        annual_vol = hist.iloc[-252:].pct_change().dropna().std() * np.sqrt(252)
        if annual_vol > MAX_ANNUAL_VOL:
            continue

        # 过滤3：200MA 过滤
        if sma200_filter:
            if len(hist) < 200:
                continue
            if hist.iloc[-1] < hist.iloc[-200:].mean():
                continue

        # vol_adjust: 动量 / 波动率 → 每单位风险的动量强度（Barroso & Santa-Clara 2015）
        scores[sym] = momentum / annual_vol if (vol_adjust and annual_vol > 0) else momentum

    return scores



def spy_above_sma200(closes, date):
    hist = closes["SPY"].loc[:date].dropna()
    if len(hist) < 200:
        return True
    return hist.iloc[-1] > hist.iloc[-200:].mean()


def macd_and_golden_cross(closes, date):
    """熔断解除：MACD 金叉 且 黄金交叉（50MA > 200MA）同时满足"""
    spy = closes["SPY"].loc[:date].dropna()
    if len(spy) < 200:
        return False
    golden = spy.iloc[-50:].mean() > spy.iloc[-200:].mean()
    if not golden:
        return False
    macd_line   = spy.ewm(span=12).mean() - spy.ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    return macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-1] > 0


# ============================================================
# 4. 回测引擎
# ============================================================

class Backtester:

    def __init__(self, closes, history_df, start, end,
                 initial_cash=INITIAL_CASH, fund_cache=None,
                 rebal_freq="weekly", vol_adjust=False):
        self.closes_full  = closes
        self.closes       = closes.loc[start:end]
        self.history_df   = history_df
        self.fund_cache   = fund_cache
        self.vol_adjust   = vol_adjust
        self.cash         = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.holdings     = {}          # {symbol: shares}
        self.nav_log      = []
        self.circuit_breaker = False
        self.peak_nav     = float(initial_cash)

        # 再平衡日历
        idx = pd.DatetimeIndex(self.closes.index)
        if rebal_freq == "monthly":
            s = pd.Series(idx, index=idx)
            self.rebal_dates = set(s.resample("ME").last().dropna())
        else:  # weekly：每周五
            self.rebal_dates = set(idx[idx.dayofweek == 4])

    def _nav(self, date):
        prices = self.closes.loc[date]
        equity = 0.0
        for sym, sh in self.holdings.items():
            p = prices.get(sym, np.nan)
            if pd.notna(p) and p > 0:
                equity += sh * p
            # 若价格为 NaN（退市），按 0 计入，避免 NaN 传播
        return self.cash + equity

    def _liquidate(self, date, symbol=None):
        """清仓全部或指定个股，扣除卖出成本"""
        prices = self.closes.loc[date]
        if symbol is not None:
            if symbol in self.holdings:
                p = prices.get(symbol, np.nan)
                if pd.notna(p) and p > 0:
                    proceeds    = self.holdings[symbol] * p
                    sell_cost   = proceeds * TRADE_COST
                    self.cash  += proceeds - sell_cost
                del self.holdings[symbol]
        else:
            for sym, sh in self.holdings.items():
                p = prices.get(sym, np.nan)
                if pd.notna(p) and p > 0:
                    proceeds   = sh * p
                    sell_cost  = proceeds * TRADE_COST
                    self.cash += proceeds - sell_cost
            self.holdings = {}

    def _buy(self, date, targets):
        """
        targets: {symbol: weight}
        只对实际发生变化的仓位收取交易成本：
          - 目标中没有的旧持仓 → 卖出（收卖出成本）
          - 目标中新增的持仓   → 买入（收买入成本）
          - 仓位不变的持仓     → 不交易，不收成本
        """
        prices = self.closes.loc[date]
        nav    = self._nav(date)

        # 计算目标持仓（股数）
        target_shares = {}
        for sym, weight in targets.items():
            p = prices.get(sym, np.nan)
            if pd.isna(p) or p <= 0:
                continue
            target_shares[sym] = (nav * weight) / p

        # 卖出：不在目标中的旧持仓
        for sym in list(self.holdings.keys()):
            if sym not in target_shares:
                p = prices.get(sym, np.nan)
                if pd.notna(p) and p > 0:
                    proceeds   = self.holdings[sym] * p
                    self.cash += proceeds * (1 - TRADE_COST)
                del self.holdings[sym]

        # 调整：在目标中的持仓，计算差额
        for sym, new_sh in target_shares.items():
            p       = prices.get(sym, np.nan)
            old_sh  = self.holdings.get(sym, 0)
            delta   = new_sh - old_sh

            if abs(delta) < 0.01:          # 差异极小，忽略不交易
                continue

            trade_value = abs(delta) * p
            if delta > 0:                  # 加仓/新买
                cost = trade_value * (1 + TRADE_COST)
                if cost > self.cash:
                    continue
                self.holdings[sym] = old_sh + delta
                self.cash         -= cost
            else:                          # 减仓/部分卖出
                proceeds           = trade_value * (1 - TRADE_COST)
                self.holdings[sym] = old_sh + delta
                self.cash         += proceeds

    def _check_drawdown(self, date):
        nav = self._nav(date)
        if nav > self.peak_nav:
            self.peak_nav = nav
        drawdown = (self.peak_nav - nav) / self.peak_nav
        if drawdown >= MAX_DRAWDOWN and not self.circuit_breaker:
            print(f"  [{date.date()}] 熔断触发！回撤 {drawdown:.2%}，转防御")
            self.circuit_breaker = True
            self._buy(date, DEFENSE)

    def _rebalance(self, date):
        # 熔断解除：MACD 金叉 且 黄金交叉
        if self.circuit_breaker:
            if macd_and_golden_cross(self.closes_full, date):
                print(f"  [{date.date()}] 熔断解除，重启进攻")
                self.circuit_breaker = False
                self.peak_nav        = self._nav(date)
                self._liquidate(date)
            else:
                return

        # 大盘防御判断
        if not spy_above_sma200(self.closes_full, date):
            print(f"  [{date.date()}] 防御模式，SPY < 200MA")
            self._buy(date, DEFENSE)
            return

        # 进攻：复合动量评分选股
        universe = get_universe_at(self.history_df, date)
        if not universe:
            print(f"  [{date.date()}] 无历史成分股数据，跳过")
            return

        stock_scores = calc_momentum(self.closes_full, date, universe,
                                     sma200_filter=True, min_up_months=MIN_UP_MONTHS,
                                     vol_adjust=self.vol_adjust)

        # 基本面过滤：ROE / 毛利率 / 营收增速（有数据才过滤，无数据则放行）
        if self.fund_cache is not None and not self.fund_cache.empty:
            fund_scores = get_fundamental_scores(
                self.fund_cache, date, list(stock_scores.keys())
            )
            filtered = {}
            for sym, mom in stock_scores.items():
                f = fund_scores.get(sym)
                if f is None:
                    filtered[sym] = mom   # 无基本面数据，放行
                    continue
                roe          = f.get("roe")
                gross_margin = f.get("gross_margin")
                rev_growth   = f.get("rev_growth")
                piotroski    = f.get("piotroski")
                if pd.notna(roe)          and roe          < MIN_ROE:
                    continue
                if pd.notna(gross_margin) and gross_margin < MIN_GROSS_MARGIN:
                    continue
                if pd.notna(rev_growth)   and rev_growth   < MIN_REV_GROWTH:
                    continue
                if pd.notna(piotroski)    and piotroski    < MIN_PIOTROSKI:
                    continue
                filtered[sym] = mom
            stock_scores = filtered

        # 动量加权排序，取前 TOP_STOCKS 只
        top_stocks = sorted(stock_scores, key=lambda x: stock_scores[x],
                            reverse=True)[:TOP_STOCKS]
        if not top_stocks:
            print(f"  [{date.date()}] 无符合条件标的，维持现金")
            return
        final_scores = np.array([stock_scores[s] for s in top_stocks])

        # 得分加权，单只上限 25%
        weights = final_scores - final_scores.min() + 1e-6
        weights = weights / weights.sum()
        weights = np.minimum(weights, 0.25)
        weights = weights / weights.sum() * STOCK_ALLOC

        targets = {s: float(w) for s, w in zip(top_stocks, weights)}
        self._buy(date, targets)
        print(f"  [{date.date()}] RISK_ON | {len(top_stocks)}只 | 前3:{top_stocks[:3]}")

    def run(self):
        print("\n开始回测...\n")
        for date in self.closes.index:
            if not self.circuit_breaker:
                self._check_drawdown(date)      # 每日：组合熔断
            if date in self.rebal_dates:
                self._rebalance(date)           # 再平衡
            self.nav_log.append({"date": date, "nav": self._nav(date)})
        return pd.DataFrame(self.nav_log).set_index("date")


# ============================================================
# 5. 绩效报告
# ============================================================

def _calc_metrics(nav: pd.Series, spy: pd.Series) -> dict:
    """计算单条净值曲线的全套绩效指标"""
    years     = (nav.index[-1] - nav.index[0]).days / 365.25
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr      = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    daily_ret = nav.pct_change().dropna()
    ann_vol   = daily_ret.std() * np.sqrt(252)
    sharpe    = daily_ret.mean() * 252 / ann_vol if ann_vol > 0 else 0
    roll_max  = nav.cummax()
    dd_series = (nav - roll_max) / roll_max
    max_dd    = dd_series.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0

    # 月度胜率
    monthly   = nav.resample("ME").last().pct_change().dropna()
    win_rate  = (monthly > 0).mean()

    # Beta / Alpha（相对 SPY）
    spy_ret  = spy.pct_change().dropna()
    common   = daily_ret.index.intersection(spy_ret.index)
    if len(common) > 60:
        beta  = daily_ret[common].cov(spy_ret[common]) / spy_ret[common].var()
        alpha = (daily_ret[common].mean() - beta * spy_ret[common].mean()) * 252
    else:
        beta, alpha = np.nan, np.nan

    spy_total = spy.iloc[-1] / spy.iloc[0] - 1
    spy_cagr  = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0

    return dict(
        years=years, total_ret=total_ret, cagr=cagr, ann_vol=ann_vol,
        sharpe=sharpe, max_dd=max_dd, calmar=calmar, win_rate=win_rate,
        beta=beta, alpha=alpha, dd_series=dd_series,
        excess_cagr=cagr - spy_cagr,
    )


def _print_table(label: str, m: dict, spy_cagr: float, spy_sharpe: float, spy_dd: float):
    print(f"\n  ── {label} ──")
    print(f"  {'总收益率':<14} {m['total_ret']:>10.2%}")
    print(f"  {'CAGR':<14} {m['cagr']:>10.2%}   SPY {spy_cagr:>7.2%}")
    print(f"  {'超额年化':<14} {m['excess_cagr']:>10.2%}")
    print(f"  {'年化波动率':<14} {m['ann_vol']:>10.2%}")
    print(f"  {'夏普比率':<14} {m['sharpe']:>10.3f}   SPY {spy_sharpe:>7.3f}")
    print(f"  {'最大回撤':<14} {m['max_dd']:>10.2%}   SPY {spy_dd:>7.2%}")
    print(f"  {'Calmar比率':<14} {m['calmar']:>10.3f}")
    print(f"  {'月度胜率':<14} {m['win_rate']:>10.1%}")
    print(f"  {'Beta':<14} {m['beta']:>10.3f}")
    print(f"  {'年化Alpha':<14} {m['alpha']:>10.2%}")


def report(nav_weekly_df: pd.DataFrame, nav_monthly_df: pd.DataFrame,
           nav_voladj_df: pd.DataFrame, closes: pd.DataFrame):
    """
    三策略对比：周度原始 / 月度原始 / 月度波动率调整 vs SPY
    图表：① 归一化净值  ② 回撤对比  ③ 日收益率直方图
    """
    nav_w = nav_weekly_df["nav"]
    nav_m = nav_monthly_df["nav"]
    nav_v = nav_voladj_df["nav"]
    spy   = closes["SPY"].reindex(nav_w.index).ffill()

    mw = _calc_metrics(nav_w, spy)
    mm = _calc_metrics(nav_m, spy.reindex(nav_m.index).ffill())
    mv = _calc_metrics(nav_v, spy.reindex(nav_v.index).ffill())

    spy_ret    = spy.pct_change().dropna()
    spy_years  = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_cagr   = (spy.iloc[-1] / spy.iloc[0]) ** (1 / spy_years) - 1
    spy_sharpe = spy_ret.mean() * 252 / (spy_ret.std() * np.sqrt(252))
    spy_dd_min = ((spy - spy.cummax()) / spy.cummax()).min()

    # ── 终端打印 ─────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"  绩效对比报告  |  成本 {TRADE_COST:.3%}/单边（长桥平台）")
    print("=" * 58)
    _print_table("周度（原始动量）",       mw, spy_cagr, spy_sharpe, spy_dd_min)
    _print_table("月度（原始动量）",       mm, spy_cagr, spy_sharpe, spy_dd_min)
    _print_table("月度（波动率调整动量）", mv, spy_cagr, spy_sharpe, spy_dd_min)
    print("=" * 58)

    # ── 颜色 ─────────────────────────────────────────────────
    C_WEEKLY  = "#2196F3"   # 蓝：周度原始
    C_MONTHLY = "#4CAF50"   # 绿：月度原始
    C_VOLADJ  = "#9C27B0"   # 紫：月度波动率调整
    C_SPY     = "#FF5722"   # 橙：SPY

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(2, 2, figure=fig, height_ratios=[3, 2], hspace=0.38, wspace=0.28)
    ax_nav  = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # ① 归一化净值
    nw = nav_w / nav_w.iloc[0]
    nm = nav_m.reindex(nav_w.index).ffill() / nav_m.iloc[0]
    nv = nav_v.reindex(nav_w.index).ffill() / nav_v.iloc[0]
    ns = spy / spy.iloc[0]

    ax_nav.plot(nw.index, nw, color=C_WEEKLY,  linewidth=1.6,
                label=f"周度·原始     CAGR {mw['cagr']:+.1%}  Sharpe {mw['sharpe']:.2f}  MaxDD {mw['max_dd']:.1%}")
    ax_nav.plot(nm.index, nm, color=C_MONTHLY, linewidth=1.8, linestyle="-.",
                label=f"月度·原始     CAGR {mm['cagr']:+.1%}  Sharpe {mm['sharpe']:.2f}  MaxDD {mm['max_dd']:.1%}")
    ax_nav.plot(nv.index, nv, color=C_VOLADJ,  linewidth=1.8, linestyle=":",
                label=f"月度·波动调整 CAGR {mv['cagr']:+.1%}  Sharpe {mv['sharpe']:.2f}  MaxDD {mv['max_dd']:.1%}")
    ax_nav.plot(ns.index, ns, color=C_SPY,     linewidth=1.1, linestyle="--", alpha=0.8,
                label=f"SPY           CAGR {spy_cagr:+.1%}  Sharpe {spy_sharpe:.2f}  MaxDD {spy_dd_min:.1%}")

    ax_nav.set_title(
        f"策略净值对比（含交易成本 {TRADE_COST:.3%}/单边）  "
        f"{nav_w.index[0].year}–{nav_w.index[-1].year}",
        fontsize=12, pad=10,
    )
    ax_nav.legend(fontsize=9, loc="upper left")
    ax_nav.set_ylabel("归一化净值（初始=1）")
    ax_nav.grid(alpha=0.25)

    # 指标文字框
    stats_text = (
        f"{'':5}{'周度原始':>9}{'月度原始':>9}{'波动调整':>9}{'SPY':>7}\n"
        f"{'CAGR':<5}{mw['cagr']:>+8.1%}{mm['cagr']:>+9.1%}{mv['cagr']:>+9.1%}{spy_cagr:>+7.1%}\n"
        f"{'超额':<5}{mw['excess_cagr']:>+8.1%}{mm['excess_cagr']:>+9.1%}{mv['excess_cagr']:>+9.1%}{'—':>7}\n"
        f"{'夏普':<5}{mw['sharpe']:>8.2f}{mm['sharpe']:>9.2f}{mv['sharpe']:>9.2f}{spy_sharpe:>7.2f}\n"
        f"{'MaxDD':<5}{mw['max_dd']:>8.1%}{mm['max_dd']:>9.1%}{mv['max_dd']:>9.1%}{spy_dd_min:>7.1%}\n"
        f"{'Calmar':<5}{mw['calmar']:>8.2f}{mm['calmar']:>9.2f}{mv['calmar']:>9.2f}{'—':>7}\n"
        f"{'月胜率':<5}{mw['win_rate']:>8.1%}{mm['win_rate']:>9.1%}{mv['win_rate']:>9.1%}{'—':>7}"
    )
    ax_nav.text(
        0.995, 0.97, stats_text,
        transform=ax_nav.transAxes, va="top", ha="right",
        fontsize=8.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92),
    )

    # ② 回撤对比
    dd_w = mw["dd_series"]
    dd_m = mm["dd_series"].reindex(dd_w.index).ffill().fillna(0)
    dd_v = mv["dd_series"].reindex(dd_w.index).ffill().fillna(0)

    ax_dd.fill_between(dd_w.index, dd_w * 100, 0, alpha=0.25, color=C_WEEKLY)
    ax_dd.fill_between(dd_m.index, dd_m * 100, 0, alpha=0.25, color=C_MONTHLY)
    ax_dd.fill_between(dd_v.index, dd_v * 100, 0, alpha=0.25, color=C_VOLADJ)
    ax_dd.plot(dd_w.index, dd_w * 100, color=C_WEEKLY,  linewidth=0.9, label="周度·原始")
    ax_dd.plot(dd_m.index, dd_m * 100, color=C_MONTHLY, linewidth=0.9, label="月度·原始",   linestyle="-.")
    ax_dd.plot(dd_v.index, dd_v * 100, color=C_VOLADJ,  linewidth=0.9, label="月度·波动调整", linestyle=":")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_title("回撤对比 (%)", fontsize=11)
    ax_dd.set_ylabel("回撤 (%)")
    ax_dd.legend(fontsize=8.5)
    ax_dd.grid(alpha=0.25)

    # ③ 日收益率直方图
    ret_w = nav_w.pct_change().dropna() * 100
    ret_m = nav_m.pct_change().dropna() * 100
    ret_v = nav_v.pct_change().dropna() * 100
    ret_s = spy.pct_change().dropna() * 100

    bins = np.linspace(-5, 5, 60)
    ax_hist.hist(ret_s, bins=bins, color=C_SPY,     alpha=0.35, label="SPY",        density=True)
    ax_hist.hist(ret_w, bins=bins, color=C_WEEKLY,  alpha=0.45, label="周度·原始",  density=True)
    ax_hist.hist(ret_m, bins=bins, color=C_MONTHLY, alpha=0.45, label="月度·原始",  density=True)
    ax_hist.hist(ret_v, bins=bins, color=C_VOLADJ,  alpha=0.45, label="月度·波动调整", density=True)

    for val, color, ls in [
        (ret_w.mean(), C_WEEKLY,  "-"),
        (ret_m.mean(), C_MONTHLY, "-."),
        (ret_v.mean(), C_VOLADJ,  ":"),
        (ret_s.mean(), C_SPY,     "--"),
    ]:
        ax_hist.axvline(val, color=color, linewidth=1.3, linestyle=ls, alpha=0.9)

    ax_hist.set_title(
        f"日收益率分布\n"
        f"月度原始 μ={ret_m.mean():.3f}% σ={ret_m.std():.2f}%   "
        f"波动调整 μ={ret_v.mean():.3f}% σ={ret_v.std():.2f}%",
        fontsize=9.5,
    )
    ax_hist.set_xlabel("日收益率 (%)")
    ax_hist.set_ylabel("概率密度")
    ax_hist.legend(fontsize=8.5)
    ax_hist.grid(alpha=0.25)

    out = "backtest_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  图表已保存：{out}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 56)
    print("  长线动量策略 v4 — 本地回测（历史成分股版）")
    print("=" * 56)

    history_df = load_sp500_history()

    if history_df is not None:
        all_universe = get_all_historical_tickers(history_df)
        print(f"历史曾入选标普500的股票共 {len(all_universe)} 只")
    else:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        resp    = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=30
        )
        table        = pd.read_html(io.StringIO(resp.text))[0]
        all_universe = table["Symbol"].str.replace(".", "-", regex=False).tolist()

    all_tickers = list(set(all_universe + list(DEFENSE.keys()) + ["SPY"]))

    closes = download_prices(all_tickers, "2007-01-01", END_DATE)
    print(f"价格数据维度：{closes.shape[0]} 天 × {closes.shape[1]} 只股票")

    # 加载基本面数据（SimFin，带缓存）
    print("\n正在加载基本面数据...")
    try:
        income, balance, cashflow = load_simfin_data()
        fund_cache                = build_fundamental_cache(income, balance, cashflow)
        print(f"基本面缓存：{len(fund_cache)} 条记录，"
              f"覆盖 {fund_cache['ticker'].nunique()} 只股票\n")
    except Exception as e:
        print(f"基本面数据加载失败（{e}），跳过基本面过滤\n")
        fund_cache = None

    print("\n── 周度回测 ──")
    bt_w  = Backtester(closes, history_df, START_DATE, END_DATE, INITIAL_CASH,
                       fund_cache=fund_cache, rebal_freq="weekly")
    nav_w = bt_w.run()

    print("\n── 月度回测（原始动量）──")
    bt_m  = Backtester(closes, history_df, START_DATE, END_DATE, INITIAL_CASH,
                       fund_cache=fund_cache, rebal_freq="monthly", vol_adjust=False)
    nav_m = bt_m.run()

    print("\n── 月度回测（波动率调整动量）──")
    bt_v  = Backtester(closes, history_df, START_DATE, END_DATE, INITIAL_CASH,
                       fund_cache=fund_cache, rebal_freq="monthly", vol_adjust=True)
    nav_v = bt_v.run()

    report(nav_w, nav_m, nav_v, closes)
