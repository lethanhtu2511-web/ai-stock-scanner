"""Research-only walk-forward smoke backtest for strategy_v2.

Uses point-in-time D1 slices and next-session execution. This is a first-pass
candidate study, not a production portfolio simulator; partial exits and
portfolio-level concurrency are reported as limitations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_provider import fetch_exchange_symbols, fetch_stock_history
from strategy_v2 import analyze_daily_setup, universe_check, _indicators

COST_BUY = 0.0015
COST_SELL = 0.0015
TAX_SELL = 0.001
SLIPPAGE_HIGH = 0.001
SLIPPAGE_LOW = 0.0025
import os

# Full current HOSE/HNX universe by default. Set V2_MAX_SYMBOLS for a bounded smoke run.
MAX_SYMBOLS = int(os.getenv("V2_MAX_SYMBOLS", "0"))  # 0 = all eligible symbols
TEST_SESSIONS = 250
TRAIN_SESSIONS = 750
HISTORY_SESSIONS = TRAIN_SESSIONS + 3 * TEST_SESSIONS
# Independent maximum-holding-period experiments. Each arm has its own
# same-symbol cooldown so outcomes are not suppressed by another horizon.
HOLDING_HORIZONS = {"2-5": (2, 5), "5-10": (5, 10), "10-20": (10, 20)}


def clean(raw):
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d and "Date" in raw.columns:
        d["date"] = raw["Date"]
    d = d.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("open","high","low","close","volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["date","open","high","low","close","volume"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def net_return(entry, exit_price, slip):
    buy = entry*(1+slip)*(1+COST_BUY)
    sell = exit_price*(1-slip)*(1-COST_SELL-TAX_SELL)
    return sell/buy-1


def simulate(d, signal_i, candidate, holding_horizon=None):
    """Conservative next-session execution with staged exits and trailing stops.

    Stop/targets are recalculated from the actual fill price. When stop and
    target are both touched in one daily candle, stop is assumed first.
    """
    nxt = signal_i + 1
    if nxt >= len(d):
        return None

    signal = d.iloc[signal_i]
    row = d.iloc[nxt]
    plan = candidate["risk_plan"]
    slip = float(plan["slippage_rate"])
    atr = float(candidate.get("atr") or 0)
    signal_close = float(signal.close)

    if candidate["entry_type"] == "NEXT_OPEN":
        raw_entry = float(row.open)
        gap_atr = abs(raw_entry - signal_close) / atr if atr > 0 else float("inf")
        if gap_atr > float(candidate["gap_limit_atr"]):
            return None
    else:
        trigger = float(candidate["entry_trigger"])
        if float(row.high) < trigger:
            return None
        # Daily OHLC data cannot establish whether the low occurred before
        # or after an intraday stop-entry trigger. If the high reaches the
        # trigger, count the fill; the stop-first check in the exit loop is the
        # conservative treatment when the same candle also touches the stop.
        # Never use the same session's closing price to erase an intraday fill.
        raw_entry = max(float(row.open), trigger)

    # Rebuild protective stop from actual fill, not signal-close reference.
    if candidate["setup"] == "PULLBACK":
        structural = float(d.iloc[max(0, signal_i-4):signal_i+1].low.min())
        stop = min(structural, raw_entry - 1.5 * atr) - 0.2 * atr
    elif candidate["setup"] == "SWING_BREAKOUT":
        structural = float(candidate.get("base_low", plan["stop_loss"]))
        stop = min(structural, raw_entry - 2.0 * atr)
    else:
        structural = float(candidate.get("confirmation_low", plan["stop_loss"] + 0.5 * atr))
        stop = structural - 0.5 * atr
    if raw_entry <= stop:
        return None
    risk = raw_entry - stop
    if risk <= 0:
        return None

    setup = candidate["setup"]
    if setup == "REVERSAL":
        target_r = 1.5
        target_2r = None
        target_3r = None
    else:
        target_r = 2.0
        target_2r = 3.0 if setup == "SWING_BREAKOUT" else None
        target_3r = None

    target1 = raw_entry + target_r * risk
    target2 = raw_entry + target_2r * risk if target_2r else None
    max_hold = int(candidate["holding_sessions"][1])
    if holding_horizon is not None:
        # In research mode, replace the setup default maximum so every setup
        # is compared under the same requested holding-period cap.
        max_hold = int(holding_horizon[1])
    last = min(len(d) - 1, nxt + max_hold - 1)

    # Indicator values are calculated using historical bars only. Trailing
    # levels for a session use the previous completed session's values.
    ind = _indicators(d.iloc[:last + 1].copy())
    remaining = 1.0
    realized = 0.0
    exits = []
    first_target_hit = False
    second_target_hit = False
    highest_high = float(row.high)
    exit_reason = "TIME_EXIT"
    exit_i = last

    def realize(fraction, price, reason, idx):
        nonlocal realized, remaining
        fraction = min(fraction, remaining)
        if fraction <= 0:
            return
        realized += fraction * net_return(raw_entry, float(price), slip)
        remaining -= fraction
        exits.append({"fraction": round(fraction, 4), "price": round(float(price), 2),
                      "reason": reason, "date": str(d.iloc[idx].date.date())})

    for j in range(nxt, last + 1):
        b = d.iloc[j]
        # Trail from information available at the prior close only.
        if j > nxt and first_target_hit:
            # The strategy rules trail only the remaining position AFTER the
            # first profit target has been reached. Before that, the original
            # protective stop remains the sole exit threshold.
            prev_ind = ind.iloc[j - 1]
            if setup in ("PULLBACK", "SWING_BREAKOUT"):
                trail = float(prev_ind.high22_prev - 3 * prev_ind.atr) if pd.notna(prev_ind.high22_prev) and pd.notna(prev_ind.atr) else stop
            else:
                trail = float(prev_ind.ema21) if pd.notna(prev_ind.ema21) else stop
            trail = max(trail, raw_entry + 0.5 * risk)
            active_stop = max(stop, trail)
        else:
            active_stop = stop

        # Conservative daily-bar ordering: stop before target if both touched.
        if float(b.open) <= active_stop:
            realize(remaining, float(b.open), "GAP_STOP", j)
            exit_i, exit_reason = j, "GAP_STOP"
            break
        if float(b.low) <= active_stop:
            realize(remaining, active_stop, "TRAIL_STOP" if active_stop > stop else "STOP", j)
            exit_i, exit_reason = j, exits[-1]["reason"]
            break

        if setup == "SWING_BREAKOUT":
            if remaining > 0 and float(b.high) >= target1:
                tranche = 1/3 if not first_target_hit else 0
                realize(tranche, target1, "TARGET_2R", j)
                first_target_hit = True
            if remaining > 0 and target2 is not None and not second_target_hit and float(b.high) >= target2:
                realize(min(1/3, remaining), target2, "TARGET_3R", j)
                second_target_hit = True
                first_target_hit = True
        else:
            if remaining > 0 and not first_target_hit and float(b.high) >= target1:
                tranche = 0.5
                realize(tranche, target1, "TARGET_1_5R" if setup == "REVERSAL" else "TARGET_2R", j)
                first_target_hit = True

        highest_high = max(highest_high, float(b.high))
        if remaining <= 1e-9:
            exit_i, exit_reason = j, "TARGETS_FILLED"
            break

    if remaining > 1e-9:
        final_price = float(d.iloc[exit_i].close)
        realize(remaining, final_price, exit_reason, exit_i)

    return {
        "entry_date": str(d.iloc[nxt].date.date()),
        "exit_date": str(d.iloc[exit_i].date.date()),
        "net_return_pct": round(realized * 100, 3),
        "win": realized > 0,
        "holding": exit_i - nxt + 1,
        "exit_reason": exit_reason,
        "score": candidate["score"],
        "exit_legs": json.dumps(exits, ensure_ascii=False),
        "actual_entry": round(raw_entry, 2),
        "initial_stop": round(stop, 2),
        "initial_risk": round(risk, 2),
        "_exit_index": int(exit_i),
    }



def point_in_time_features(d, signal_i, market_ind, market_i, trade):
    r = d.iloc[signal_i]
    close = float(r.close)
    atr = float(r.atr) if pd.notna(r.atr) else 0.0
    ema21 = float(r.ema21) if pd.notna(r.ema21) else None
    out = {
        "signal_rsi": float(r.rsi) if pd.notna(r.rsi) else None,
        "signal_adx": float(r.adx) if pd.notna(r.adx) else None,
        "signal_vol_ratio": float(r.vol_ratio) if pd.notna(r.vol_ratio) else None,
        "signal_atr_pct": atr / close * 100 if close > 0 and atr > 0 else None,
        "signal_close_to_ema21_atr": (close-ema21)/atr if ema21 is not None and atr > 0 else None,
    }
    for n in (20, 60):
        if signal_i >= n and market_i >= n:
            sr = close / float(d.iloc[signal_i-n].close) - 1
            mr = float(market_ind.iloc[market_i].close) / float(market_ind.iloc[market_i-n].close) - 1
            out[f"relative_strength_{n}s_pct"] = (sr-mr)*100
        else:
            out[f"relative_strength_{n}s_pct"] = None
    entry, stop = float(trade.get("actual_entry") or 0), float(trade.get("initial_stop") or 0)
    out["signal_entry_gap_pct"] = (entry/close-1)*100 if close > 0 and entry > 0 else None
    out["signal_stop_distance_pct"] = (entry-stop)/entry*100 if entry > stop > 0 else None
    return {k: round(v, 4) if v is not None else None for k, v in out.items()}


def swing_filter_diagnostics(frame):
    if frame.empty:
        return {}
    swing = frame[frame.setup == "SWING_BREAKOUT"].copy()
    if swing.empty:
        return {}
    rs20 = pd.to_numeric(swing["relative_strength_20s_pct"], errors="coerce")
    rs60 = pd.to_numeric(swing["relative_strength_60s_pct"], errors="coerce")
    vol = pd.to_numeric(swing["signal_vol_ratio"], errors="coerce")
    adx = pd.to_numeric(swing["signal_adx"], errors="coerce")
    gap = pd.to_numeric(swing["signal_entry_gap_pct"], errors="coerce")
    stop = pd.to_numeric(swing["signal_stop_distance_pct"], errors="coerce")
    masks = {
        "BASELINE": pd.Series(True, index=swing.index),
        "RS20_POSITIVE": rs20 > 0,
        "RS60_POSITIVE": rs60 > 0,
        "RS20_AND_RS60_POSITIVE": (rs20 > 0) & (rs60 > 0),
        "VOLUME_RATIO_GE_1_5": vol >= 1.5,
        "ADX_GE_20": adx >= 20,
        "RS20_AND_VOLUME_GE_1_5": (rs20 > 0) & (vol >= 1.5),
        "RS20_AND_ADX_GE_20": (rs20 > 0) & (adx >= 20),
        "RS20_AND_ENTRY_GAP_LE_0_5": (rs20 > 0) & (gap <= 0.5),
        "RS20_AND_STOP_DISTANCE_LE_8": (rs20 > 0) & (stop <= 8),
    }
    def compact(sub):
        v = pd.to_numeric(sub.net_return_pct, errors="coerce").dropna()
        gains, losses = float(v[v > 0].sum()), abs(float(v[v < 0].sum()))
        return {"trades": int(len(v)),
                "win_rate_pct": round(float((v > 0).mean()*100), 2) if len(v) else None,
                "average_net_return_pct": round(float(v.mean()), 3) if len(v) else None,
                "median_net_return_pct": round(float(v.median()), 3) if len(v) else None,
                "profit_factor": round(gains/losses, 3) if losses > 0 else None}
    result = {}
    for name, mask in masks.items():
        subset = swing.loc[mask]
        item = compact(subset)
        item["by_oos_window"] = {str(int(w)): compact(g) for w, g in subset.groupby("oos_window")}
        item["warning"] = "Descriptive subgroup only; not a validated trading rule."
        result[name] = item
    return result


def main():
    listed=sorted(set(fetch_exchange_symbols()))
    symbols = listed if MAX_SYMBOLS == 0 else listed[:MAX_SYMBOLS]
    market=clean(fetch_stock_history("VNINDEX",length=HISTORY_SESSIONS,is_index=True))
    ranked=[]
    histories={}
    errors=[]
    progress=Path("strategy_v2_output"); progress.mkdir(exist_ok=True)
    for number, s in enumerate(symbols, 1):
        try:
            d=clean(fetch_stock_history(s,length=HISTORY_SESSIONS))
            if len(d) < TRAIN_SESSIONS + TEST_SESSIONS:
                errors.append({"symbol":s,"error":f"Chỉ có {len(d)} phiên; cần ít nhất {TRAIN_SESSIONS + TEST_SESSIONS}."})
                continue
            cutoff=TRAIN_SESSIONS
            eligible, blockers=universe_check(d.iloc[:cutoff+1])
            if not eligible:
                continue
            avg=float((d.close*d.volume).iloc[:cutoff+1].tail(20).mean())
            # Precompute the complete indicator matrix once per symbol. The
            # walk-forward loop below slices only past rows, avoiding O(n^2)
            # repeated EMA/ADX/RSI calculations for every historical session.
            histories[s]=_indicators(d)
            ranked.append((s,avg))
        except Exception as exc:
            errors.append({"symbol":s,"error":str(exc)[:180]})
        if number % 10 == 0 or number == len(symbols):
            (progress/"progress.json").write_text(json.dumps({
                "symbols_processed":number,"symbols_total":len(symbols),
                "downloaded_symbols":len(histories),"eligible_symbols":[x[0] for x in ranked],
                "errors":errors[-100:]
            },ensure_ascii=False,indent=2),encoding="utf-8")
    ranked.sort(key=lambda x:x[1],reverse=True)
    chosen=[s for s,_ in ranked]
    if MAX_SYMBOLS > 0:
        chosen=chosen[:MAX_SYMBOLS]
    trades=[]
    diagnostics={k:{"evaluated":0,"candidates":0,"blocked_regime":0} for k in ("PULLBACK","SWING_BREAKOUT","REVERSAL")}
    # Precompute VNINDEX indicators and resolve its regime by date using
    # searchsorted; never rebuild the index indicators inside the stock loop.
    market_ind = _indicators(market)
    market_dates = pd.to_datetime(market_ind.date).to_numpy()
    sessions_done = 0
    for symbol_number, s in enumerate(chosen, 1):
        d=histories[s]
        next_free_by_horizon={name: TRAIN_SESSIONS for name in HOLDING_HORIZONS}
        for i in range(TRAIN_SESSIONS,len(d)-1):
            asof=d.iloc[:i+1]
            asof_date=pd.Timestamp(asof.iloc[-1].date).to_datetime64()
            market_i=int(pd.Series(market_dates).searchsorted(asof_date, side="right"))-1
            if market_i < 199:
                continue
            mr=market_ind.iloc[market_i]
            if pd.isna(mr.sma50) or pd.isna(mr.sma200):
                regime="UNKNOWN"
            elif mr.close < mr.sma200:
                regime="BEARISH"
            elif mr.sma50 > mr.sma200:
                regime="BULLISH"
            else:
                regime="NEUTRAL"
            res=analyze_daily_setup(s,asof,None,indicators_precomputed=True,
                                    market_regime_override=regime)
            sessions_done += 1
            for setup in diagnostics:
                diagnostics[setup]["evaluated"]+=1
                if res.get("market_regime")=="BEARISH":
                    diagnostics[setup]["blocked_regime"]+=1
            candidates=res.get("candidates",[])
            for c in candidates:
                diagnostics[c["setup"]]["candidates"]+=1
            if not candidates:
                continue
            c=candidates[0]
            for horizon_name, horizon_range in HOLDING_HORIZONS.items():
                if i < next_free_by_horizon[horizon_name]:
                    continue
                trade=simulate(d,i,c,holding_horizon=horizon_range)
                if trade:
                    next_free_by_horizon[horizon_name]=trade.pop("_exit_index")+1
                    trade["oos_window"]=(i-TRAIN_SESSIONS)//TEST_SESSIONS+1
                    trade["holding_horizon"]=horizon_name
                    features = point_in_time_features(d, i, market_ind, market_i, trade)
                    trades.append({"symbol":s,"setup":c["setup"],**trade,**features})
            if sessions_done % 100 == 0:
                (progress/"progress.json").write_text(json.dumps({
                    "phase":"walk_forward","symbols_processed":symbol_number,
                    "symbols_total":len(chosen),"current_symbol":s,
                    "sessions_evaluated":sessions_done,"trades_found":len(trades),
                    "diagnostics":diagnostics
                },ensure_ascii=False,indent=2),encoding="utf-8")
        (progress/"progress.json").write_text(json.dumps({
            "phase":"walk_forward","symbols_processed":symbol_number,
            "symbols_total":len(chosen),"current_symbol":s,
            "sessions_evaluated":sessions_done,"trades_found":len(trades),
            "diagnostics":diagnostics
        },ensure_ascii=False,indent=2),encoding="utf-8")
    frame=pd.DataFrame(trades)
    if frame.empty:
        frame=pd.DataFrame(columns=["symbol","setup","net_return_pct","win","oos_window","holding_horizon"])
    def summarize(group):
        values=pd.to_numeric(group.net_return_pct,errors="coerce").dropna()
        wins=float(values[values>0].sum())
        losses=abs(float(values[values<0].sum()))
        return {"trades":int(len(values)),
            "win_rate_pct":round(float((values>0).mean()*100),2) if len(values) else 0,
            "avg_net_return_pct":round(float(values.mean()),3) if len(values) else 0,
            "median_net_return_pct":round(float(values.median()),3) if len(values) else 0,
            "profit_factor":round(wins/losses,3) if losses>0 else None,
            "total_compounded_pct":round(float(((1+values/100).prod()-1)*100),3) if len(values) else 0}
    horizon_performance={}
    for horizon_name in HOLDING_HORIZONS:
        hg=frame[frame.holding_horizon==horizon_name] if not frame.empty else frame
        horizon_performance[horizon_name]={setup:summarize(hg[hg.setup==setup]) for setup in diagnostics}
    # Preserve the legacy performance key as the shortest 2-5 session arm.
    summary=horizon_performance["2-5"]
    window_summary={}
    if not frame.empty:
        for window, wg in frame.groupby("oos_window"):
            window_summary[str(int(window))]={}
            for horizon_name in HOLDING_HORIZONS:
                hg=wg[wg.holding_horizon==horizon_name]
                window_summary[str(int(window))][horizon_name]={
                    setup:summarize(hg[hg.setup==setup]) for setup in diagnostics
                }
    out=Path("strategy_v2_output"); out.mkdir(exist_ok=True)
    frame.to_csv(out/"trades.csv",index=False)
    report={"engine":"PRECOMPUTED_INDICATORS_WALK_FORWARD",
        "sessions_evaluated":sessions_done,
        "universe_mode":"ALL_HOSE_HNX" if MAX_SYMBOLS == 0 else "BOUNDED",
        "symbols_found_on_exchange":len(symbols),"symbols_tested":len(chosen),"symbols_eligible":len(ranked),
        "market_data_end":str(market.date.max().date()),"test_sessions_per_window":TEST_SESSIONS,"train_sessions":TRAIN_SESSIONS,
        "oos_windows": max([max(0,(len(histories[s])-TRAIN_SESSIONS)//TEST_SESSIONS) for s in chosen],default=0),
        "oos_window_performance":window_summary,
        "diagnostics":diagnostics,"performance":summary,
        "performance_by_holding_horizon":horizon_performance,
        "swing_filter_diagnostics": swing_filter_diagnostics(frame[frame.holding_horizon=="2-5"]),
        "holding_horizons": {name:{"minimum_sessions":bounds[0],"maximum_sessions":bounds[1],
            "note":"Independent research arm; maximum hold is set to the horizon upper bound for every setup; stop/targets and costs unchanged."}
            for name,bounds in HOLDING_HORIZONS.items()},
        "data_errors":errors[:100],
        "costs":{"buy_fee":COST_BUY,"sell_fee":COST_SELL,"sell_tax":TAX_SELL,
                 "slippage": "0.10% high-liquidity; 0.25% low-liquidity"},
        "limitations":["Signals use current listing universe; delisted historical constituents are not reconstructed (survivorship bias remains).",
           "Each horizon arm independently limits each symbol to one active trade; signals are evaluated point-in-time and overlapping same-symbol trades are suppressed within each arm. The same signal stream is replayed independently across horizons.",
           "Holding-horizon arms change only the maximum time stop (2-5, 5-10, 10-20 sessions); entry, protective stop, targets, next-session execution, fees, taxes and slippage are held constant.",
           "Daily-bar simulation uses stop-first assumption when stop and target occur in the same candle.",
           "Fixed rules are not fitted; the 3-year segment is indicator warm-up/history followed by out-of-sample evaluation.",
           "Industry concentration, portfolio-wide cash allocation, daily circuit breaker, drawdown throttle and loss-streak throttle are not yet simulated."]}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not chosen:
        raise RuntimeError("No eligible symbols; no strategy result can be inferred.")


if __name__=="__main__":
    main()

# Expanded-universe validation rerun requested on 2026-11-21; keep MAX_SYMBOLS=0 for all listed HOSE/HNX symbols.
