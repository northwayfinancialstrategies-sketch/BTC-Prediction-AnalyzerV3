"""BTC Prediction Analyzer v3 — public, read-only research and paper trading.

Prices: Kalshi public asks. BTC price: Coinbase Exchange spot reference.
The optional volatility scenario is UNVALIDATED and must never be used as
an independent estimate of a profitable trading advantage.
"""

import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

API = "https://external-api.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com"
EASTERN = ZoneInfo("America/New_York")
MOUNTAIN = ZoneInfo("America/Denver")
EVENT_RE = re.compile(r"KXBTCD-\d{2}[A-Z]{3}\d{4}", re.I)
NORMAL = statistics.NormalDist()
HEADERS = {"User-Agent": "BTC-Prediction-Analyzer/3.1", "Accept": "application/json"}


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def parse_time(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def parse_event(value):
    m = EVENT_RE.search(str(value).strip())
    return m.group(0).upper() if m else None


def event_time(event):
    try:
        return datetime.strptime(event.removeprefix("KXBTCD-"), "%y%b%d%H").replace(tzinfo=EASTERN)
    except (ValueError, TypeError):
        return None


def next_event():
    upcoming = datetime.now(EASTERN).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return f"KXBTCD-{upcoming.strftime('%y%b%d%H').upper()}"


def define_contract(m):
    """Use explicit Kalshi strike fields; never infer an unknown market's threshold."""
    kind = str(m.get("strike_type") or "").lower()
    floor, cap = number(m.get("floor_strike")), number(m.get("cap_strike"))
    if kind == "between" and floor is not None and cap is not None and cap > floor:
        return f"BTC between ${floor:,.2f} and ${cap:,.2f}", "between", floor, cap
    if kind in ("less", "less_or_equal") and cap is not None:
        sign = "<" if kind == "less" else "≤"
        return f"BTC {sign} ${cap:,.2f}", kind, cap, None
    if kind in ("greater", "greater_or_equal") and floor is not None:
        sign = ">" if kind == "greater" else "≥"
        return f"BTC {sign} ${floor:,.2f}", kind, floor, None
    # A few legacy API responses omit strike_type; only use unambiguous floor-only strikes.
    if not kind and floor is not None and cap is None:
        return f"BTC above ${floor:,.2f} (check exact rules)", "greater", floor, None
    title = m.get("yes_sub_title") or m.get("title") or m.get("ticker") or "Unknown contract"
    return str(title)[:90], "unsupported", None, None


def public_ask(m, side):
    val = number(m.get(f"{side.lower()}_ask_dollars"))
    if val is None:
        cents = number(m.get(f"{side.lower()}_ask"))
        val = cents / 100 if cents is not None else None
    return val if val is not None and 0 < val < 1 else None


def quote_size(m, side):
    return number(m.get(f"{side.lower()}_ask_size_fp"))


def has_quoted_size(budget, fee, ask, available):
    if ask is None or budget <= fee:
        return "Unavailable"
    if available is None:
        return "Size not reported"
    needed = (budget - fee) / ask
    return "Size sufficient at snapshot" if available >= needed else "Size too small for stake"


def buffer(spot, kind, floor, cap, side):
    if spot is None or floor is None:
        return None
    if kind in ("greater", "greater_or_equal"):
        distance = spot - floor
    elif kind in ("less", "less_or_equal"):
        distance = floor - spot
    elif kind == "between" and cap is not None:
        distance = min(spot - floor, cap - spot)
    else:
        return None
    return distance if side == "YES" else -distance


def probability_yes(spot, kind, floor, cap, sigma):
    """Illustrative zero-drift lognormal scenario, NOT an outcome-calibrated forecast."""
    if spot is None or spot <= 0 or floor is None or floor <= 0 or sigma is None or sigma <= 0:
        return None
    if kind in ("greater", "greater_or_equal"):
        p = 1 - NORMAL.cdf(math.log(floor / spot) / sigma)
    elif kind in ("less", "less_or_equal"):
        p = NORMAL.cdf(math.log(floor / spot) / sigma)
    elif kind == "between" and cap is not None and cap > floor:
        p = NORMAL.cdf(math.log(cap / spot) / sigma) - NORMAL.cdf(math.log(floor / spot) / sigma)
    else:
        return None
    return max(0.0, min(1.0, p))


def payout(budget, fee, ask):
    """Illustration allowing fractional contracts; actual Coinbase checkout overrides."""
    if ask is None or budget <= fee or fee < 0:
        return None
    units = (budget - fee) / ask
    gross = units  # $1 per winning binary contract
    return gross, gross - budget, budget / gross, units


@st.cache_data(ttl=25, show_spinner=False)
def get_markets(event):
    result, cursor, visited = [], None, set()
    for _ in range(12):
        params = {"event_ticker": event, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{API}/markets", params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        result.extend(payload.get("markets", []))
        cursor = payload.get("cursor")
        if not cursor or cursor in visited:
            break
        visited.add(cursor)
    unique = {m["ticker"]: m for m in result if m.get("ticker")}
    return list(unique.values())


@st.cache_data(ttl=25, show_spinner=False)
def get_spot():
    resp = requests.get(f"{COINBASE}/products/BTC-USD/ticker", headers=HEADERS, timeout=12)
    resp.raise_for_status()
    payload = resp.json()
    return float(payload["price"]), payload.get("time", "")


@st.cache_data(ttl=120, show_spinner=False)
def get_candles():
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=23, minutes=30)
    resp = requests.get(
        f"{COINBASE}/products/BTC-USD/candles",
        params={"granularity": 300, "start": start.isoformat(), "end": end.isoformat()},
        headers=HEADERS, timeout=15,
    )
    resp.raise_for_status()
    bars = resp.json()
    if not isinstance(bars, list):
        raise ValueError("Coinbase candle response was not a list")
    closes = sorted(
        [(int(b[0]), float(b[4])) for b in bars if len(b) >= 5 and number(b[4])],
        key=lambda x: x[0],
    )
    changes = []
    for (t0, p0), (t1, p1) in zip(closes, closes[1:]):
        if 240 <= t1 - t0 <= 360 and p0 > 0 and p1 > 0:
            changes.append(math.log(p1 / p0))
    return (statistics.stdev(changes), len(changes)) if len(changes) >= 80 else (None, len(changes))


def get_final_result(ticker):
    """Look up exchange-reported YES/NO determination for this EXACT contract."""
    if not re.fullmatch(r"[A-Z0-9_.-]{8,160}", str(ticker)):
        raise ValueError("Invalid contract ticker")
    resp = requests.get(f"{API}/markets/{ticker}", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    market = resp.json()["market"]
    if market.get("ticker") != ticker:
        raise ValueError("Received a different contract from the one requested")
    value = str(market.get("result") or "").lower()
    return value if value in ("yes", "no") else None


def classify_row(low_gap, high_gap, quote_status):
    """Paper-research categories ONLY, not an investment recommendation."""
    if quote_status == "Size too small for stake":
        return "Insufficient quoted size"
    if low_gap is None or high_gap is None:
        return "Unrated — no model"
    if high_gap < 0:
        return "Negative in all scenarios"
    if low_gap <= 0:
        return "Depends on assumptions"
    return "Model-only positive — UNVALIDATED"


st.set_page_config(page_title="BTC Prediction Analyzer v3", page_icon="₿", layout="wide")
st.title("₿ BTC Prediction Analyzer · v3")
st.caption("Public Kalshi contract data + Coinbase Exchange reference price · research / paper tests only")
st.markdown("**Simple workflow:** Choose next hour → confirm exact strike + YES/NO → save paper trade → verify Kalshi result after settlement. **Do not buy based on the model gap.**")
st.error("**No real-money recommendation is generated here.** Scenario probabilities have NOT been validated against settlements. A positive 'model gap' does not establish a profitable trade.")

with st.sidebar:
    st.header("Your settings")
    stake = st.number_input("Research stake ($)", 1.0, 100000.0, 50.0, 5.0)
    fees = st.number_input("Known total fees ($), if any", 0.0, 1000.0, 0.0, 0.25)
    limit = st.number_input("Total amount you allow yourself to lose ($)", 1.0, 100000.0, 50.0, 5.0)
    how_many = st.slider("Number of nearest contracts in main view", 5, 50, 15)
    sort_by = st.selectbox("Sort table by", ["Largest unvalidated model gap", "Nearest to BTC price"])
    st.caption("Does not connect to Coinbase funds or place orders.")
if stake > limit:
    st.error("Your proposed stake exceeds your stated loss limit. Reduce it before evaluating trades.")
if fees >= stake:
    st.error("Known fees cannot be equal to or greater than the stake.")
    st.stop()

st.subheader("1 · Choose the hourly market")
mode = st.radio("Market", ["Next hour automatically", "Paste Coinbase event URL / ticker"], horizontal=True)
if mode.startswith("Next"):
    event = next_event()
else:
    raw = st.text_input("Coinbase market link or event ID", value="https://www.coinbase.com/predictions/event/KXBTCD-26NOV2213")
    event = parse_event(raw)
if not event:
    st.error("No valid KXBTCD event ID found in that link.")
    st.stop()
if st.button("↻ Refresh public prices and BTC reference"):
    get_markets.clear()
    get_spot.clear()
    get_candles.clear()
    st.rerun()

scheduled = event_time(event)
st.code(event, language=None)
if scheduled:
    st.caption(f"Scheduled event: {scheduled.strftime('%b %d %Y, %I:%M %p ET')} · {scheduled.astimezone(MOUNTAIN).strftime('%I:%M %p MT')} (from event ID, not official market closing time)")

try:
    spot, spot_time = get_spot()
    st.metric("BTC reference price (Coinbase Exchange spot; NOT the Kalshi settlement index)", f"${spot:,.2f}")
    if spot_time:
        st.caption(f"Coinbase spot timestamp: {spot_time}")
except (requests.RequestException, ValueError, KeyError) as exc:
    spot = None
    st.warning(f"BTC reference unavailable: {type(exc).__name__}")
reference = st.number_input("BTC spot used for illustration ($); enter manually if needed", 0.0, 1000000.0, float(spot or 0), 10.0, format="%.2f")
reference = reference if reference > 0 else None

try:
    markets = get_markets(event)
except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
    markets = []
    st.error(f"Kalshi markets could not be retrieved: {type(exc).__name__}. Try the refresh button.")

try:
    sd_5min, sample_n = get_candles()
except (requests.RequestException, ValueError, TypeError):
    sd_5min, sample_n = None, 0

st.subheader("2 · How to read the table")
st.markdown("**Start with `Model gap (pp)`** = experimental scenario chance **minus** the required break-even chance. Bigger positive numbers are worth **paper-testing**, not buying. Check the **sensitivity range**, **quoted size**, and **exact contract ID** before recording anything.")
if sd_5min:
    st.caption(f"Experimental model uses {sample_n} recent valid 5-minute BTC returns. Zero drift, lognormal returns, volatility multiplied by 0.7×–1.3×. NOT backtested and NOT a confidence interval.")
else:
    st.info("Not enough valid Coinbase candles for a model. Break-even math and public quotes can still be shown.")

rows = []
now = datetime.now(timezone.utc)
for market in markets:
    if str(market.get("status") or "").lower() not in ("active", "open"):
        continue
    label, typ, floor, cap = define_contract(market)
    close = parse_time(market.get("close_time"))
    minutes = (close - now).total_seconds() / 60 if close else None
    updated = parse_time(market.get("updated_time"))
    age_min = max(0.0, (now - updated).total_seconds()/60) if updated else None
    for side in ("YES", "NO"):
        ask = public_ask(market, side)
        if ask is None:
            continue
        size = quote_size(market, side)
        size_label = has_quoted_size(stake, fees, ask, size)
        math_result = payout(stake, fees, ask)
        if math_result is None:
            continue
        gross, profit, breakeven, units = math_result
        # No estimated payout for $50 at the displayed ask if the first quoted size is too small.
        if size_label == "Size too small for stake":
            gross, profit = None, None
        scenario, p_lo, p_hi = None, None, None
        if reference and sd_5min and minutes and minutes > 0 and typ != "unsupported":
            sigma = sd_5min * math.sqrt(minutes / 5)
            ps = [probability_yes(reference, typ, floor, cap, sigma * mult) for mult in (0.7, 1, 1.3)]
            if all(p is not None for p in ps):
                if side == "NO":
                    ps = [1 - p for p in ps]
                p_lo, p_hi = min(ps), max(ps)
                scenario = ps[1]
        base_gap = 100 * (scenario - breakeven) if scenario is not None else None
        low_gap = 100 * (p_lo - breakeven) if p_lo is not None else None
        high_gap = 100 * (p_hi - breakeven) if p_hi is not None else None
        assessment = classify_row(low_gap, high_gap, size_label)
        # Kalshi updated_time is a market-record timestamp, NOT necessarily the quote time.
        # Never label a quote stale solely from this field.
        rows.append({
            "Contract": label, "Side": side, "Ask ($)": ask,
            "Break-even (%)": 100 * breakeven,
            "Scenario chance (%)": 100 * scenario if scenario is not None else None,
            "Model gap (pp)": base_gap,
            "Gap range low (pp)": low_gap, "Gap range high (pp)": high_gap,
            "Public payout ($)": gross, "Public profit ($)": profit,
            "BTC buffer ($)": buffer(reference, typ, floor, cap, side),
            "Quote size": size_label,
            "Quote age (min)": age_min,
            "Research status": assessment,
            "Market closes UTC": close.isoformat() if close else "",
            "ticker": market["ticker"], "strike_type": typ,
            "floor_strike": floor, "cap_strike": cap,
            "event": event, "fee": fees, "stake": stake,
            "units_needed": units, "ask_size": size,
            "minutes_left": minutes,
            "sort_distance": abs(floor - reference) if floor is not None and reference else float("inf"),
        })

st.subheader("3 · Actual public quotes + MODEL GAP")
st.caption("Kalshi asks can be stale, too small or unavailable at Coinbase's final checkout price. The model gap is a hypothetical difference, NOT a verified trading edge. Public payouts approximate fractional contracts and known manual fees only.")
selected = None
if rows:
    df = pd.DataFrame(rows)
    closest = df.sort_values("sort_distance")["ticker"].drop_duplicates().head(how_many)
    shown = df[df["ticker"].isin(closest)].copy()
    if sort_by.startswith("Largest"):
        shown = shown.sort_values(["Model gap (pp)", "sort_distance"], ascending=[False, True], na_position="last")
    else:
        shown = shown.sort_values(["sort_distance", "ticker", "Side"])
    display_columns = ["Model gap (pp)", "Scenario chance (%)", "Break-even (%)", "Contract", "Side",
                       "Ask ($)", "Public profit ($)", "BTC buffer ($)", "Quote size", "Research status"]
    st.dataframe(
        shown[display_columns], hide_index=True, use_container_width=True,
        column_config={
            "Model gap (pp)": st.column_config.NumberColumn(help="Unvalidated scenario minus break-even in percentage points", format="%+.1f"),
            "Scenario chance (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Break-even (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Ask ($)": st.column_config.NumberColumn(format="$%.4f"),
            "Public profit ($)": st.column_config.NumberColumn(format="$%.2f"),
            "BTC buffer ($)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(f"{len(markets)} contracts found in this event; showing {len(shown)} YES/NO quotes from the nearest {min(how_many, len(closest))} strike contracts. Sorting by model gap is for PAPER RESEARCH only.")
    st.download_button("Download ALL quoted rows (CSV)",
                       data=df.drop(columns=["sort_distance"], errors="ignore").to_csv(index=False).encode("utf-8"),
                       file_name=f"{event}_research_quotes.csv", mime="text/csv")
    st.info("**Interpretation:** Positive model gap = only that the untested model looks more optimistic than the quoted break-even rate. It cannot demonstrate value. An old quote or insufficient size is not a price you can assume you could buy.")

    # Make the first option the nearest *displayed* contract rather than API order.
    # Users can explicitly expand to all contracts; never silently preselect a distant strike.
    show_all_choices = st.checkbox("Allow selection from ALL contracts (not just nearby strikes)", value=False)
    selectable = rows if show_all_choices else shown.to_dict(orient="records")
    selectable = sorted(selectable, key=lambda r: (float("inf") if r.get("sort_distance") is None else r["sort_distance"], r["ticker"], r["Side"]))
    choices = {f"{r['Contract']} · {r['Side']} · {r['ticker']}": r for r in selectable}
    chosen_label = st.selectbox("4 · Inspect one EXACT contract", list(choices), key=f"selected_contract_{event}_{show_all_choices}")
    selected = choices[chosen_label]
    st.markdown(f"**LOCKED-IN SELECTION TO REVIEW:** `{selected['ticker']}` · **{selected['Side']}** · {selected['Contract']}")
    if selected["Market closes UTC"]:
        close_dt = parse_time(selected["Market closes UTC"])
        st.write(f"**Kalshi market close:** {close_dt.astimezone(MOUNTAIN).strftime('%b %d, %I:%M %p MT')} · {max(0.0, (close_dt-now).total_seconds()/60):.1f} minutes remaining")
    else:
        st.warning("Official market close time missing — do not record until confirmed.")
    if selected["Public payout ($)"] is None:
        st.error("Insufficient size at the quoted ask for your full stake. No public full-stake payout is shown.")
    else:
        st.write(f"**Illustrative public payout:** ${selected['Public payout ($)']:,.2f} total · **profit if correct:** ${selected['Public profit ($)']:,.2f} · **loss if wrong:** ${stake:,.2f}")
    if selected["Scenario chance (%)"] is not None:
        st.write(f"**Experimental model:** {selected['Scenario chance (%)']:.1f}% versus **{selected['Break-even (%)']:.1f}% break-even** = **{selected['Model gap (pp)']:+.1f} percentage-point gap**. Sensitivity: {selected['Gap range low (pp)']:+.1f} to {selected['Gap range high (pp)']:+.1f} points.")
    st.warning(f"**Research status: {selected['Research status']}** — This is not a buy recommendation.")
else:
    st.info("No active contracts with usable public ask prices for this hourly event. Try the next upcoming event.")

st.subheader("5 · Coinbase final checkout calculator (separate from public data)")
st.caption("Only use the total Coinbase says **'to win'** at checkout. No purchase is needed to see a quote; Coinbase may differ from public Kalshi asks.")
checkout = st.number_input("Actual Coinbase total payout if correct ($); 0 = unknown", min_value=0.0, value=0.0, step=1.0)
if checkout > 0:
    c1, c2, c3 = st.columns(3)
    c1.metric("Profit if correct", f"${checkout - stake:,.2f}")
    c2.metric("Loss if incorrect", f"-${stake:,.2f}")
    c3.metric("Actual checkout break-even", f"{100 * stake / checkout:.1f}%")
    st.caption("Assumes this is your final full payout and your entire amount at risk is the stake entered; do not subtract the same fee twice.")

st.subheader("6 · Paper-trade log — confirm exact side and contract")
st.caption("This app NEVER places a trade. The selection is saved as a snapshot; never assume a selected row is the same contract you viewed earlier.")
if "paper_log_v3" not in st.session_state:
    st.session_state.paper_log_v3 = []

incoming = st.file_uploader("Optional: import an earlier paper journal CSV (including v2 exports)", type=["csv"])
if incoming is not None:
    file_bytes = incoming.getvalue()
    file_signature = (incoming.name, len(file_bytes), hash(file_bytes))
    if st.session_state.get("last_paper_import") != file_signature:
        try:
            old = pd.read_csv(incoming).fillna("")
            new_rows = old.to_dict(orient="records")
            existing = {(str(x.get("observed_utc")), str(x.get("ticker")), str(x.get("side"))) for x in st.session_state.paper_log_v3}
            added = 0
            for r in new_rows:
                key = (str(r.get("observed_utc")), str(r.get("ticker")), str(r.get("side")))
                if key not in existing:
                    st.session_state.paper_log_v3.append(r)
                    existing.add(key)
                    added += 1
            st.session_state.last_paper_import = file_signature
            st.success(f"Imported {added} additional saved rows. Review older tickers and sides before relying on them.")
        except (ValueError, UnicodeError, pd.errors.ParserError):
            st.error("That CSV could not be parsed. Please upload a journal exported by this app.")

if selected:
    exact = f"{selected['ticker']} · {selected['Side']} · {selected['Contract']}"
    st.code(exact, language=None)
    confirmation = st.checkbox("I checked the EXACT ticker, strike, YES/NO side and closing time shown above", key=f"confirm_{selected['ticker']}_{selected['Side']}")
    ready = bool(selected["Market closes UTC"]) and bool(selected["minutes_left"] and selected["minutes_left"] > 0)
    if not ready:
        st.warning("Recording disabled because this contract's official close time is missing or has passed.")
    if st.button("Record EXACT selected contract (PAPER ONLY)", disabled=not (confirmation and ready)):
        st.session_state.paper_log_v3.append({
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "event": event, "ticker": selected["ticker"], "contract": selected["Contract"],
            "strike_type": selected["strike_type"], "floor_strike": selected["floor_strike"],
            "cap_strike": selected["cap_strike"], "side": selected["Side"],
            "market_close_utc": selected["Market closes UTC"],
            "reference_spot": reference, "stake": stake,
            "public_ask": selected["Ask ($)"], "quoted_size": selected["ask_size"],
            "quote_coverage": selected["Quote size"],
            "public_gross_payout": selected["Public payout ($)"],
            "public_profit_if_win": selected["Public profit ($)"],
            "fee_estimate": fees,
            "model_scenario_pct_UNVALIDATED": selected["Scenario chance (%)"],
            "model_gap_pp_UNVALIDATED": selected["Model gap (pp)"],
            "checkout_total_payout_if_entered": checkout or "",
            "outcome": "pending", "result_source": "", "paper_pnl_illustrative": "",
        })
        st.success(f"Saved PAPER selection: {exact}. Download the CSV to preserve it.")

if st.session_state.paper_log_v3:
    log_df = pd.DataFrame(st.session_state.paper_log_v3)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    options = [f"{i+1} · {r.get('ticker', 'unknown')} · {r.get('side', 'UNKNOWN')} · {r.get('observed_utc', '')}" for i, r in enumerate(st.session_state.paper_log_v3)]
    idx = st.selectbox("Choose a saved paper row to verify AFTER settlement", range(len(options)), format_func=lambda i: options[i])
    if st.button("Check selected row's exchange-reported YES/NO result"):
        record = st.session_state.paper_log_v3[idx]
        ticker = str(record.get("ticker") or "")
        side = str(record.get("side") or "").upper()
        if side not in ("YES", "NO"):
            st.error("Saved row does not contain a valid YES/NO side. Correct your original selection before assessing it.")
        else:
            try:
                official = get_final_result(ticker)
                if official is None:
                    st.info("Kalshi has not reported a final YES/NO result for this exact ticker yet. Keep it pending.")
                else:
                    won = official.upper() == side
                    record["outcome"] = "WIN" if won else "LOSS"
                    record["result_source"] = "Kalshi API market.result"
                    stake_saved = number(record.get("stake"))
                    actual_payout = number(record.get("checkout_total_payout_if_entered"))
                    public_payout = number(record.get("public_gross_payout"))
                    if stake_saved is not None:
                        # Only indicative cash P&L: this is not executed Coinbase trade data.
                        if won and (actual_payout or public_payout):
                            record["paper_pnl_illustrative"] = round((actual_payout or public_payout) - stake_saved, 2)
                        elif not won:
                            record["paper_pnl_illustrative"] = -stake_saved
                    st.success(f"Exchange result for {ticker}: {official.upper()}. Your saved {side} paper side: {'WIN' if won else 'LOSS'}.")
            except (requests.RequestException, KeyError, ValueError) as exc:
                st.error(f"Could not verify the outcome from Kalshi ({type(exc).__name__}). No result was guessed.")
    if st.button("Verify pending paper outcomes (up to 20 per click)"):
        checked = resolved = failed = 0
        for record in st.session_state.paper_log_v3:
            if checked >= 20:
                break
            if record.get("result_source") == "Kalshi API market.result":
                continue
            ticker = str(record.get("ticker") or "")
            side = str(record.get("side") or "").upper()
            close_at = parse_time(record.get("market_close_utc"))
            if side not in ("YES", "NO") or not re.fullmatch(r"[A-Z0-9_.-]{8,160}", ticker):
                continue
            if close_at is not None and close_at > datetime.now(timezone.utc):
                continue
            checked += 1
            try:
                final = get_final_result(ticker)
                if final is None:
                    continue
                won = final.upper() == side
                record["outcome"] = "WIN" if won else "LOSS"
                record["result_source"] = "Kalshi API market.result"
                cost = number(record.get("stake"))
                paid = number(record.get("checkout_total_payout_if_entered"))
                if paid is None or paid <= 0:
                    paid = number(record.get("public_gross_payout"))
                if cost is not None:
                    if won and paid is not None and paid > 0:
                        record["paper_pnl_illustrative"] = round(paid - cost, 2)
                    elif not won:
                        record["paper_pnl_illustrative"] = -cost
                resolved += 1
            except (requests.RequestException, ValueError, KeyError):
                failed += 1
        st.info(f"Checked {checked} pending rows: {resolved} exchange-verified, {failed} API errors; remaining rows stay pending. Download the new CSV below.")
    updated = pd.DataFrame(st.session_state.paper_log_v3)
    st.download_button("Download paper journal CSV (keep this copy)",
                       updated.to_csv(index=False).encode("utf-8"),
                       file_name="btc_prediction_paper_journal_v3.csv", mime="text/csv")
    st.caption("Paper P&L uses recorded *indicative* prices or a manually entered checkout quote, never a claim of actual executed profit. Earlier imported selections need independent confirmation.")
    # Do not treat manually typed WIN/LOSS or user-reported spot as exchange-verified.
    confirmed = [r for r in st.session_state.paper_log_v3 if r.get("result_source") == "Kalshi API market.result" and r.get("outcome") in ("WIN", "LOSS")]
    st.markdown("**Verified-only paper performance**")
    if not confirmed:
        st.info("No Kalshi-verified outcomes yet. Select a saved row above and click 'Check selected row's exchange-reported YES/NO result' after settlement.")
    else:
        unique_events = {(str(r.get("event")), str(r.get("ticker")), str(r.get("side"))) for r in confirmed}
        wins = sum(r.get("outcome") == "WIN" for r in confirmed)
        pnls = [number(r.get("paper_pnl_illustrative")) for r in confirmed]
        usable = [v for v in pnls if v is not None]
        m1, m2, m3 = st.columns(3)
        m1.metric("Verified paper outcomes", f"{wins} wins / {len(confirmed) - wins} losses")
        m2.metric("Illustrative P&L with recorded quotes", f"${sum(usable):+,.2f}" if len(usable) == len(confirmed) else "Not fully known")
        m3.metric("Distinct ticker/side combinations", str(len(unique_events)))
        st.caption("Repeated records of one contract count as separate time snapshots; verify independence before interpreting aggregate accuracy. This is not real profit.")

with st.expander("Why this is NOT a trading signal"):
    st.markdown("""
    - **Public ask ≠ true event probability.** It is a potential buying cost and may include spread; displayed quote size can be insufficient. Actual Coinbase checkout is the authority for your possible cost.
    - **Scenario probability is unvalidated.** A 0.7×–1.3× volatility test is not a confidence interval. BTC can move much more than a normal distribution suggests.
    - **A positive model gap is not evidence of an edge.** It is only a paper-research filter until a long, out-of-sample study shows calibration after fees.
    - **The Coinbase Exchange reference spot is NOT the CF Benchmarks settlement index.** Check the exchange-reported result using the exact contract ticker after settlement.
    - **The paper log saves exactly the displayed ticker and side** at the time you press Record. Existing older CSV rows may reflect different contracts or chosen sides; verify each one.
    - The app is **read-only** and cannot place orders, access wallets, or guarantee income.
    """)
st.caption("Sources: Kalshi public Trade API markets and market-result endpoints; Coinbase Exchange public BTC ticker and candles. No authenticated trading calls.")
