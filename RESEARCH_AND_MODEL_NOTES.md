# Research notes — AI Stock Scanner

## Evidence used
- Nguyen et al. (2021), *Does short-term technical trading exist in the Vietnamese stock market?* Reports moderate statistical-arbitrage opportunities in the VN30 sample after transaction costs; momentum models worked for specific stocks, not uniformly.
- Metghalchi (2013), *Market Efficiency and Profitability of Technical Trading Rules: Evidence from Vietnam*. Historical VN-Index study of moving averages and other technical rules; findings are sample- and period-dependent.
- Shinhan Securities Vietnam (15 Apr 2026), public investor education on volume: compare relative volume to the stock's own history; assess breakout with price structure, closing location and subsequent hold/retest; abnormal volume alone is not a buy signal.
- SSI FastConnect documentation describes REST and streaming data, but SSI's own onboarding guide says registration is required. Do not assume anonymous/free access.
- TCBS OpenAPI documentation describes WebSocket market data but requires API key/OTP authentication. Verify account eligibility and commercial terms before integrating.

## Model design principles
1. Separate early setup alerts from confirmed buy decisions.
2. Compare volume with the same symbol's baseline, not absolute volume across symbols.
3. Require price structure, trend, liquidity and market-regime context together.
4. Use ATR for volatility-aware stops and position sizing; include fees, tax and slippage.
5. Avoid look-ahead bias: only use completed bars for historical features; live quotes must be timestamp-verified.
6. Backtest chronologically with realistic execution assumptions, transaction costs, slippage, delisted symbols and survivorship-bias checks.
7. Report expectancy, profit factor, max drawdown, win rate, average holding period, turnover and number of trades. Do not present score as win probability.

## Caveats
The public literature and investor education are inputs, not proof of future profitability. No universal "best" indicator exists. Strategy thresholds must be calibrated out-of-sample on the intended universe and trading horizon.
