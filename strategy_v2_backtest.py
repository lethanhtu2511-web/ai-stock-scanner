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
from strategy_v2 import analyze_daily_setup, market_regime, universe_check

COST_BUY = 0.0015
COST_SELL = 0.0015
TAX_SELL = 0.001
SLIPPAGE_HIGH = 0.001
SLIPPAGE_LOW = 0.0025
MAX_SYMBOLS = 10
PILOT_SYMBOLS = ["FPT", "HPG", "SSI", "MBB", "CTG", "VCB", "VNM", "MWG", "TCB", "GAS"]
TEST_SESSIONS = 250


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


def simulate(d, signal_i, candidate):
    nxt = signal_i+1
    if nxt >= len(d):
        return None
    row = d.iloc[nxt]
    slip = candidate["risk_plan"]["slippage_rate"]
    if candidate["entry_type"] == "NEXT_OPEN":
        raw_entry = float(row.open)
        gap = abs(raw_entry-float(d.iloc[signal_i].close))/float(candidate["risk_plan"]["R"])
        if raw_entry > float(d.iloc[signal_i].close) and gap > candidate["gap_limit_atr"]*float(candidate["risk_plan"]["R"]):
            return None
    else:
        trigger = float(candidate["entry_trigger"])
        if float(row.high) < trigger:
            return None
        raw_entry = max(float(row.open), trigger)
        if float(row.close) < float(candidate["cancel_if_close_below_confirmation_low"]):
            return None
    entry = raw_entry*(1+slip)
    stop = float(candidate["risk_plan"]["stop_loss"])
    target = float(candidate["risk_plan"]["target_1"])
    max_hold = int(candidate["holding_sessions"][1])
    last = min(len(d)-1, nxt+max_hold-1)
    exit_price, exit_i, reason = float(d.iloc[last].close), last, "TIME_EXIT"
    for j in range(nxt,last+1):
        b=d.iloc[j]
        if float(b.open) <= stop:
            exit_price,exit_i,reason=float(b.open),j,"GAP_STOP"; break
        if float(b.low) <= stop:
            exit_price,exit_i,reason=stop,j,"STOP"; break
        if float(b.high) >= target:
            exit_price,exit_i,reason=target,j,"TARGET_1"; break
    ret=net_return(raw_entry,exit_price,slip)
    return {"entry_date":str(d.iloc[nxt].date.date()),"exit_date":str(d.iloc[exit_i].date.date()),
            "net_return_pct":round(ret*100,3),"win":ret>0,"holding":exit_i-nxt+1,
            "exit_reason":reason,"score":candidate["score"]}


def main():
    listed=set(fetch_exchange_symbols())
    symbols=[s for s in PILOT_SYMBOLS if s in listed]
    missing=[s for s in PILOT_SYMBOLS if s not in listed]
    market=clean(fetch_stock_history("VNINDEX",length=900,is_index=True))
    ranked=[]
    histories={}
    errors=[{"symbol":s,"error":"Không có trong danh sách HOSE/HNX hiện tại."} for s in missing]
    for s in symbols:
        try:
            d=clean(fetch_stock_history(s,length=900))
            if len(d)<500:
                errors.append({"symbol":s,"error":f"Chỉ lấy được {len(d)} phiên; cần 500 phiên cho warm-up + 250 phiên kiểm thử."})
                continue
            # Universe selection from the beginning of the held-out interval.
            cutoff=max(250,len(d)-TEST_SESSIONS-30)
            eligible,_=universe_check(d.iloc[:cutoff+1])
            if not eligible:
                continue
            avg=float((d.close*d.volume).iloc[:cutoff+1].tail(20).mean())
            histories[s]=d
            ranked.append((s,avg))
            progress=Path("strategy_v2_output"); progress.mkdir(exist_ok=True)
            (progress/"progress.json").write_text(json.dumps({
                "downloaded_symbols": list(histories.keys()),
                "eligible_symbols": [x[0] for x in ranked],
                "errors": errors
            },ensure_ascii=False,indent=2),encoding="utf-8")
        except Exception as exc:
            errors.append({"symbol":s,"error":str(exc)[:180]})
    ranked.sort(key=lambda x:x[1],reverse=True)
    chosen=[s for s,_ in ranked[:MAX_SYMBOLS]]
    trades=[]
    diagnostics={k:{"evaluated":0,"candidates":0,"blocked_regime":0} for k in ("PULLBACK","SWING_BREAKOUT","REVERSAL")}
    for s in chosen:
        d=histories[s]
        start=max(250,len(d)-TEST_SESSIONS-30)
        for i in range(start,len(d)-1):
            asof=d.iloc[:i+1].copy()
            mkt=market.loc[market.date<=asof.iloc[-1].date].copy()
            if len(mkt)<200:
                continue
            res=analyze_daily_setup(s,asof,mkt)
            for setup in diagnostics:
                diagnostics[setup]["evaluated"]+=1
                if res.get("market_regime")=="BEARISH":
                    diagnostics[setup]["blocked_regime"]+=1
            for c in res.get("candidates",[]):
                diagnostics[c["setup"]]["candidates"]+=1
                trade=simulate(d,i,c)
                if trade:
                    trades.append({"symbol":s,"setup":c["setup"],**trade})
    frame=pd.DataFrame(trades)
    summary={}
    for setup in diagnostics:
        g=frame[frame.setup==setup] if not frame.empty else frame
        if len(g):
            wins=g[g.net_return_pct>0].net_return_pct.sum()
            losses=abs(g[g.net_return_pct<0].net_return_pct.sum())
            summary[setup]={"trades":len(g),"win_rate_pct":round(g.win.mean()*100,2),
                "avg_net_return_pct":round(g.net_return_pct.mean(),3),
                "profit_factor":round(wins/losses,3) if losses else None,
                "total_compounded_pct":round(((1+g.net_return_pct/100).prod()-1)*100,3)}
        else:
            summary[setup]={"trades":0,"win_rate_pct":0,"avg_net_return_pct":0,"profit_factor":None}
    out=Path("strategy_v2_output"); out.mkdir(exist_ok=True)
    frame.to_csv(out/"trades.csv",index=False)
    report={"pilot_symbols_requested":PILOT_SYMBOLS,"symbols_found_on_exchange":symbols,
        "symbols_tested":len(chosen),"symbols_eligible":len(ranked),
        "market_data_end":str(market.date.max().date()),"test_sessions":TEST_SESSIONS,
        "diagnostics":diagnostics,"performance":summary,"data_errors":errors[:100],
        "costs":{"buy_fee":COST_BUY,"sell_fee":COST_SELL,"sell_tax":TAX_SELL,
                 "slippage": "0.10% high-liquidity; 0.25% low-liquidity"},
        "limitations":["Initial pilot research only; 10-symbol cap (FPT, HPG, SSI, MBB, CTG, VCB, VNM, MWG, TCB, GAS).",
          "Single target exit proxy; partial take-profit and Chandelier trailing not yet simulated.",
          "No portfolio concurrency, industry cap, daily circuit breaker or walk-forward 3y/1y aggregation yet.",
          "Current listing universe can create survivorship bias."]}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not chosen:
        raise RuntimeError("No eligible symbols; no strategy result can be inferred.")


if __name__=="__main__":
    main()
