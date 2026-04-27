# ============================================================
# 基本面数据层（SimFin 扩展版）
#
# 提取 13 个点位正确的因子，覆盖四大维度：
#   盈利能力：ROE, ROA, 毛利率, 营业利润率, 净利润率
#   成长性：  营收增速YoY, 利润增速YoY, 营收加速度
#   现金流质量：FCF利润率, 盈余质量(OCF/净利润)
#   财务健康：负债率, 流动比率
#   综合质量：Piotroski F-Score (0–9)
#
# 点位正确（无前视偏差）：
#   全部使用 Publish Date <= 当前再平衡日期的财报
#   TTM = 最近4个季报之和
#
# 附带 IC 验证辅助函数：calc_ic_series()
# ============================================================

import os
import numpy as np
import pandas as pd
import simfin as sf
from scipy import stats
from typing import Optional

SIMFIN_API_KEY = "867fdd4a-f6ce-4fb2-94db-aff1698d82d5"
SIMFIN_DIR     = "./cache/simfin"
CACHE_FILE     = f"{SIMFIN_DIR}/fundamental_cache_v2.pkl"


# ============================================================
# 内部工具
# ============================================================

def _setup():
    os.makedirs(SIMFIN_DIR, exist_ok=True)
    sf.set_api_key(SIMFIN_API_KEY)
    sf.set_data_dir(SIMFIN_DIR)


def _ttm(series: pd.Series) -> float:
    """最近4个季度之和（TTM）"""
    vals = series.dropna().iloc[-4:]
    return vals.sum() if len(vals) == 4 else np.nan


def _safe_div(num, den) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return num / den


# ============================================================
# 数据加载
# ============================================================

def load_simfin_data():
    """
    下载并缓存美股季报三张表。
    首次运行约 10-15 分钟，之后走本地缓存。
    返回：(income, balance, cashflow)
    """
    _setup()
    print("正在加载损益表（季报）...")
    income = sf.load_income(variant="quarterly", market="us")

    print("正在加载资产负债表（季报）...")
    balance = sf.load_balance(variant="quarterly", market="us")

    print("正在加载现金流量表（季报）...")
    cashflow = sf.load_cashflow(variant="quarterly", market="us")

    print(f"损益表：{len(income)} 条  资产负债表：{len(balance)} 条  "
          f"现金流量表：{len(cashflow)} 条")
    return income, balance, cashflow


# ============================================================
# 因子计算
# ============================================================

def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """从候选列名列表中找到第一个存在的列"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_fundamental_cache(income: pd.DataFrame,
                            balance: pd.DataFrame,
                            cashflow: pd.DataFrame) -> pd.DataFrame:
    """
    构造按（Ticker, Publish Date）索引的因子缓存表。
    每行代表"在该发布日期之后可见的最新 TTM 指标"。

    输出列：
      ticker, publish_date,
      roe, roa, gross_margin, op_margin, net_margin,
      rev_growth, earnings_growth, rev_accel,
      fcf_margin, earnings_quality,
      debt_ratio, current_ratio,
      piotroski
    """
    _setup()
    if os.path.exists(CACHE_FILE):
        print("从缓存加载基本面因子...")
        return pd.read_pickle(CACHE_FILE)

    print("正在计算基本面因子（首次较慢，约15-30分钟）...")

    # ── 列名映射（SimFin 官方列名 → 内部名） ──────────────────
    def _prep_income(df):
        df = df.reset_index()
        col = {
            "Ticker":       "ticker",
            "Publish Date": "publish_date",
        }
        rev_c   = _find_col(df, ["Revenue"])
        gp_c    = _find_col(df, ["Gross Profit"])
        op_c    = _find_col(df, ["Operating Income (Loss)", "Operating Income"])
        ni_c    = _find_col(df, ["Net Income", "Net Income (Common)"])
        for src, dst in [(rev_c,"revenue"),(gp_c,"gross_profit"),
                         (op_c,"op_income"),(ni_c,"net_income")]:
            if src:
                col[src] = dst
        keep = [k for k in col if k in df.columns]
        return df[keep].rename(columns=col)

    def _prep_balance(df):
        df = df.reset_index()
        col = {
            "Ticker":       "ticker",
            "Publish Date": "publish_date",
        }
        ta_c  = _find_col(df, ["Total Assets"])
        te_c  = _find_col(df, ["Total Equity", "Common Equity", "Shareholders Equity"])
        tl_c  = _find_col(df, ["Total Liabilities"])
        ca_c  = _find_col(df, ["Total Current Assets"])
        cl_c  = _find_col(df, ["Total Current Liabilities"])
        for src, dst in [(ta_c,"total_assets"),(te_c,"equity"),
                         (tl_c,"total_liab"),(ca_c,"current_assets"),
                         (cl_c,"current_liab")]:
            if src:
                col[src] = dst
        keep = [k for k in col if k in df.columns]
        return df[keep].rename(columns=col)

    def _prep_cashflow(df):
        df = df.reset_index()
        col = {
            "Ticker":       "ticker",
            "Publish Date": "publish_date",
        }
        ocf_c  = _find_col(df, ["Net Cash from Operating Activities",
                                 "Cash from Operating Activities"])
        capex_c = _find_col(df, ["Acquisition of Property, Plant & Equipment",
                                  "Capital Expenditures",
                                  "Purchase of Property, Plant & Equipment"])
        for src, dst in [(ocf_c,"ocf"),(capex_c,"capex")]:
            if src:
                col[src] = dst
        keep = [k for k in col if k in df.columns]
        return df[keep].rename(columns=col)

    inc = _prep_income(income)
    bal = _prep_balance(balance)
    cf  = _prep_cashflow(cashflow)

    for d in [inc, bal, cf]:
        d["publish_date"] = pd.to_datetime(d["publish_date"])
        d.sort_values(["ticker", "publish_date"], inplace=True)

    tickers = inc["ticker"].unique()
    total   = len(tickers)
    records = []

    for i, ticker in enumerate(tickers):
        if i % 200 == 0:
            print(f"  处理中 {i}/{total}...")

        inc_t = inc[inc["ticker"] == ticker].set_index("publish_date")
        bal_t = bal[bal["ticker"] == ticker].set_index("publish_date")
        cf_t  = cf[cf["ticker"] == ticker].set_index("publish_date") if len(cf) > 0 else pd.DataFrame()

        for pub_date in inc_t.index:
            inc_p = inc_t[inc_t.index <= pub_date]
            bal_p = bal_t[bal_t.index <= pub_date]
            cf_p  = cf_t[cf_t.index <= pub_date] if not cf_t.empty else pd.DataFrame()

            if len(inc_p) < 4 or bal_p.empty:
                continue

            # ── TTM 值 ───────────────────────────────────────
            rev_ttm = _ttm(inc_p.get("revenue",     pd.Series(dtype=float)))
            gp_ttm  = _ttm(inc_p.get("gross_profit", pd.Series(dtype=float)))
            op_ttm  = _ttm(inc_p.get("op_income",   pd.Series(dtype=float)))
            ni_ttm  = _ttm(inc_p.get("net_income",  pd.Series(dtype=float)))
            ocf_ttm = _ttm(cf_p.get("ocf",  pd.Series(dtype=float))) if not cf_p.empty else np.nan
            cap_ttm = _ttm(cf_p.get("capex", pd.Series(dtype=float))) if not cf_p.empty else np.nan

            latest_bal = bal_p.iloc[-1]
            total_assets  = latest_bal.get("total_assets",   np.nan)
            equity        = latest_bal.get("equity",         np.nan)
            total_liab    = latest_bal.get("total_liab",     np.nan)
            current_assets = latest_bal.get("current_assets", np.nan)
            current_liab   = latest_bal.get("current_liab",   np.nan)

            fcf_ttm = (ocf_ttm + cap_ttm) if pd.notna(ocf_ttm) and pd.notna(cap_ttm) else np.nan

            # ── 盈利能力 ──────────────────────────────────────
            roe          = _safe_div(ni_ttm, equity)
            roa          = _safe_div(ni_ttm, total_assets)
            gross_margin = _safe_div(gp_ttm, rev_ttm)
            op_margin    = _safe_div(op_ttm, rev_ttm)
            net_margin   = _safe_div(ni_ttm, rev_ttm)

            # ── 成长性 ────────────────────────────────────────
            cutoff_1y  = pub_date - pd.DateOffset(years=1)
            inc_1y     = inc_p[inc_p.index <= cutoff_1y]
            rev_1y     = _ttm(inc_1y.get("revenue",    pd.Series(dtype=float))) if len(inc_1y) >= 4 else np.nan
            ni_1y      = _ttm(inc_1y.get("net_income", pd.Series(dtype=float))) if len(inc_1y) >= 4 else np.nan
            rev_growth = _safe_div(rev_ttm - rev_1y, abs(rev_1y)) if pd.notna(rev_1y) else np.nan
            ni_growth  = _safe_div(ni_ttm  - ni_1y,  abs(ni_1y))  if pd.notna(ni_1y)  else np.nan

            # 营收加速度：当期增速 - 上期增速（需要2年数据）
            cutoff_2y  = pub_date - pd.DateOffset(years=2)
            inc_2y     = inc_p[inc_p.index <= cutoff_2y]
            rev_2y     = _ttm(inc_2y.get("revenue", pd.Series(dtype=float))) if len(inc_2y) >= 4 else np.nan
            rev_growth_lag = _safe_div(rev_1y - rev_2y, abs(rev_2y)) if pd.notna(rev_2y) and pd.notna(rev_1y) else np.nan
            rev_accel  = (rev_growth - rev_growth_lag) if pd.notna(rev_growth) and pd.notna(rev_growth_lag) else np.nan

            # ── 现金流质量 ────────────────────────────────────
            fcf_margin       = _safe_div(fcf_ttm, rev_ttm)
            earnings_quality = _safe_div(ocf_ttm, ni_ttm)   # > 1 说明利润有现金支撑

            # ── 财务健康 ──────────────────────────────────────
            debt_ratio    = _safe_div(total_liab, total_assets)
            current_ratio = _safe_div(current_assets, current_liab)

            # ── Piotroski F-Score（9个信号，每个0/1） ─────────
            piotroski = _calc_piotroski(
                roa=roa, ocf_ttm=ocf_ttm, ni_ttm=ni_ttm,
                total_assets=total_assets,
                inc_p=inc_p, bal_p=bal_p, cf_p=cf_p,
                rev_ttm=rev_ttm,
                debt_ratio=debt_ratio,
                current_ratio=current_ratio,
                equity=equity,
                pub_date=pub_date,
            )

            records.append({
                "ticker":           ticker,
                "publish_date":     pub_date,
                # 盈利能力
                "roe":              roe,
                "roa":              roa,
                "gross_margin":     gross_margin,
                "op_margin":        op_margin,
                "net_margin":       net_margin,
                # 成长性
                "rev_growth":       rev_growth,
                "earnings_growth":  ni_growth,
                "rev_accel":        rev_accel,
                # 现金流质量
                "fcf_margin":       fcf_margin,
                "earnings_quality": earnings_quality,
                # 财务健康
                "debt_ratio":       debt_ratio,
                "current_ratio":    current_ratio,
                # 综合质量
                "piotroski":        piotroski,
            })

    df = pd.DataFrame(records)
    df.to_pickle(CACHE_FILE)
    print(f"基本面因子缓存完成：{len(df)} 条记录，"
          f"覆盖 {df['ticker'].nunique()} 只股票")
    return df


def _calc_piotroski(roa, ocf_ttm, ni_ttm, total_assets,
                    inc_p, bal_p, cf_p, rev_ttm,
                    debt_ratio, current_ratio, equity,
                    pub_date) -> int:
    """
    Piotroski F-Score：9个会计信号，各0/1，总分0-9。
    分组：盈利(3) + 杠杆/流动性(3) + 效率(3)
    """
    score = 0

    # ── 盈利信号 ─────────────────────────────────────────────
    # F1：ROA > 0
    if pd.notna(roa) and roa > 0:
        score += 1

    # F2：经营现金流 > 0
    if pd.notna(ocf_ttm) and ocf_ttm > 0:
        score += 1

    # F3：ROA 同比改善
    cutoff_1y = pub_date - pd.DateOffset(years=1)
    inc_1y    = inc_p[inc_p.index <= cutoff_1y]
    bal_1y    = bal_p[bal_p.index <= cutoff_1y]
    if len(inc_1y) >= 4 and not bal_1y.empty:
        ni_1y  = _ttm(inc_1y.get("net_income", pd.Series(dtype=float)))
        ta_1y  = bal_1y.iloc[-1].get("total_assets", np.nan)
        roa_1y = _safe_div(ni_1y, ta_1y)
        if pd.notna(roa) and pd.notna(roa_1y) and roa > roa_1y:
            score += 1

    # F4：盈余质量（OCF > 净利润）
    if pd.notna(ocf_ttm) and pd.notna(ni_ttm) and ocf_ttm > ni_ttm:
        score += 1

    # ── 杠杆/流动性信号 ───────────────────────────────────────
    # F5：负债率下降
    if not bal_1y.empty:
        tl_1y = bal_1y.iloc[-1].get("total_liab",   np.nan)
        ta_1y_val = bal_1y.iloc[-1].get("total_assets", np.nan)
        dr_1y = _safe_div(tl_1y, ta_1y_val)
        if pd.notna(debt_ratio) and pd.notna(dr_1y) and debt_ratio < dr_1y:
            score += 1

    # F6：流动比率改善
    if not bal_1y.empty:
        ca_1y = bal_1y.iloc[-1].get("current_assets", np.nan)
        cl_1y = bal_1y.iloc[-1].get("current_liab",   np.nan)
        cr_1y = _safe_div(ca_1y, cl_1y)
        if pd.notna(current_ratio) and pd.notna(cr_1y) and current_ratio > cr_1y:
            score += 1

    # F7：未增发新股（股东权益相对总资产未明显扩大，简化处理）
    if not bal_1y.empty:
        eq_1y = bal_1y.iloc[-1].get("equity", np.nan)
        ta_1y_val2 = bal_1y.iloc[-1].get("total_assets", np.nan)
        eq_ratio_now = _safe_div(equity,  total_assets)
        eq_ratio_1y  = _safe_div(eq_1y,  ta_1y_val2)
        if pd.notna(eq_ratio_now) and pd.notna(eq_ratio_1y) and eq_ratio_now >= eq_ratio_1y:
            score += 1

    # ── 效率信号 ──────────────────────────────────────────────
    # F8：毛利率改善
    if len(inc_1y) >= 4:
        rev_1y_val = _ttm(inc_1y.get("revenue",      pd.Series(dtype=float)))
        gp_1y_val  = _ttm(inc_1y.get("gross_profit", pd.Series(dtype=float)))
        gm_1y      = _safe_div(gp_1y_val, rev_1y_val)
        gp_now     = _ttm(inc_p.get("gross_profit", pd.Series(dtype=float)))
        gm_now     = _safe_div(gp_now, rev_ttm)
        if pd.notna(gm_now) and pd.notna(gm_1y) and gm_now > gm_1y:
            score += 1

    # F9：资产周转率改善
    at_now = _safe_div(rev_ttm, total_assets)
    if len(inc_1y) >= 4 and not bal_1y.empty:
        rev_1y2 = _ttm(inc_1y.get("revenue", pd.Series(dtype=float)))
        ta_1y3  = bal_1y.iloc[-1].get("total_assets", np.nan)
        at_1y   = _safe_div(rev_1y2, ta_1y3)
        if pd.notna(at_now) and pd.notna(at_1y) and at_now > at_1y:
            score += 1

    return score


# ============================================================
# 截面查询
# ============================================================

def get_fundamental_scores(fund_cache: pd.DataFrame,
                           as_of: pd.Timestamp,
                           tickers: list) -> dict:
    """
    获取指定日期可见的最新基本面因子。
    只使用 publish_date <= as_of 的财报（无前视偏差）。

    返回：{ticker: {roe, roa, gross_margin, op_margin, net_margin,
                    rev_growth, earnings_growth, rev_accel,
                    fcf_margin, earnings_quality,
                    debt_ratio, current_ratio, piotroski}}
    """
    cutoff = fund_cache[fund_cache["publish_date"] <= as_of]
    latest = (
        cutoff.sort_values("publish_date")
              .groupby("ticker")
              .last()
              .reset_index()
    )
    ticker_set = set(tickers)
    result = {}
    for _, row in latest.iterrows():
        t = row["ticker"]
        if t in ticker_set:
            result[t] = row.drop(["ticker", "publish_date"]).to_dict()
    return result


# ============================================================
# IC 验证辅助
# ============================================================

FACTOR_COLS = [
    "roe", "roa", "gross_margin", "op_margin", "net_margin",
    "rev_growth", "earnings_growth", "rev_accel",
    "fcf_margin", "earnings_quality",
    "debt_ratio", "current_ratio", "piotroski",
]


def calc_ic_series(fund_cache: pd.DataFrame,
                   closes: pd.DataFrame,
                   factor_col: str,
                   forward_days: int = 20,
                   rebal_freq: str = "ME") -> pd.Series:
    """
    计算某因子随时间的截面 IC（Rank IC = Spearman 相关系数）。

    参数：
      fund_cache   : build_fundamental_cache() 返回的 DataFrame
      closes       : 价格 DataFrame（日期 × Ticker）
      factor_col   : 要验证的因子列名（见 FACTOR_COLS）
      forward_days : 预测窗口（交易日数），默认 20（约1个月）
      rebal_freq   : 截面频率，默认每月末（"ME"）

    返回：pd.Series，index=截面日期，values=该截面 IC
    """
    assert factor_col in fund_cache.columns, f"未知因子列：{factor_col}"

    dates = pd.date_range(
        start=fund_cache["publish_date"].min(),
        end=closes.index[-1] - pd.Timedelta(days=forward_days * 2),
        freq=rebal_freq,
    )

    ic_records = {}
    for dt in dates:
        # 当前截面因子值
        cutoff = fund_cache[fund_cache["publish_date"] <= dt]
        latest = (
            cutoff.sort_values("publish_date")
                  .groupby("ticker")
                  .last()
                  .reset_index()
        )[["ticker", factor_col]].dropna()

        if len(latest) < 10:
            continue

        # 未来 forward_days 收益率
        idx = closes.index
        pos_now  = idx.searchsorted(dt, side="right") - 1
        pos_fwd  = pos_now + forward_days
        if pos_now < 0 or pos_fwd >= len(idx):
            continue

        date_now = idx[pos_now]
        date_fwd = idx[pos_fwd]

        fwd_ret = (closes.loc[date_fwd] / closes.loc[date_now] - 1).dropna()
        fwd_ret.index.name = "ticker"
        fwd_ret = fwd_ret.reset_index()
        fwd_ret.columns = ["ticker", "fwd_return"]

        merged = latest.merge(fwd_ret, on="ticker")
        if len(merged) < 10:
            continue

        ic, _ = stats.spearmanr(merged[factor_col], merged["fwd_return"])
        if not np.isnan(ic):
            ic_records[dt] = ic

    return pd.Series(ic_records, name=factor_col)


def run_ic_report(fund_cache: pd.DataFrame,
                  closes: pd.DataFrame,
                  forward_days: int = 20) -> pd.DataFrame:
    """
    对所有 FACTOR_COLS 跑 IC 验证，打印汇总报告。

    返回：DataFrame，每行一个因子，列为 IC均值/ICIR/IC>0占比
    """
    rows = []
    for col in FACTOR_COLS:
        ic_s = calc_ic_series(fund_cache, closes, col, forward_days)
        if ic_s.empty:
            continue
        rows.append({
            "factor":    col,
            "IC均值":    ic_s.mean(),
            "IC标准差":  ic_s.std(),
            "ICIR":      ic_s.mean() / ic_s.std() if ic_s.std() > 0 else np.nan,
            "IC>0占比":  (ic_s > 0).mean(),
            "样本数":    len(ic_s),
        })

    report = pd.DataFrame(rows).sort_values("ICIR", ascending=False)

    print("\n" + "=" * 62)
    print(f"  因子 IC 验证报告（前向 {forward_days} 交易日）")
    print("=" * 62)
    print(f"  {'因子':<20} {'IC均值':>8} {'ICIR':>8} {'IC>0占比':>10} {'评级':>6}")
    print("-" * 62)
    for _, r in report.iterrows():
        ic_mean = r["IC均值"]
        icir    = r["ICIR"]
        if abs(ic_mean) > 0.05 and abs(icir) > 0.5:
            grade = "★★★"
        elif abs(ic_mean) > 0.03 or abs(icir) > 0.3:
            grade = "★★"
        else:
            grade = "★"
        print(f"  {r['factor']:<20} {ic_mean:>8.4f} {icir:>8.3f} "
              f"{r['IC>0占比']:>10.1%} {grade:>6}")
    print("=" * 62)
    print("  评级标准：★★★ |IC|>0.05 且 |ICIR|>0.5（显著有效）")
    print("           ★★  |IC|>0.03 或 |ICIR|>0.3（弱有效）")
    print("           ★   噪音为主，不建议使用")
    return report


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    income, balance, cashflow = load_simfin_data()

    print("\n损益表列名：",    income.reset_index().columns.tolist()[:15])
    print("资产负债表列名：", balance.reset_index().columns.tolist()[:15])
    print("现金流列名：",     cashflow.reset_index().columns.tolist()[:15])

    cache = build_fundamental_cache(income, balance, cashflow)
    print(cache.head(3).to_string())

    # 截面查询示例
    scores = get_fundamental_scores(cache, pd.Timestamp("2023-01-31"),
                                    ["AAPL", "MSFT", "NVDA"])
    for ticker, s in scores.items():
        print(f"\n{ticker}:")
        for k, v in s.items():
            print(f"  {k:<20} {v:.3f}" if pd.notna(v) else f"  {k:<20} NaN")
