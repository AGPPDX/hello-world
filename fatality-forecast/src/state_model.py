"""State-level forecasts of traffic fatalities, 2025-2029, for the 50 states + DC.

Usage: python src/state_model.py data
Reads  data/fars_state_annual.csv, data/state_vmt.csv, data/raw/states/*.csv, data/results.json (national blend)
Writes data/state_results.json and data/state_forecasts.csv

Method
- Seven models per state, each scored with the same rolling backtest as the national model
  (train through each year 2005..2019, forecast the next 1-5 years):
    naive, 3-year average, damped trend, ARIMA, rate-per-person x population,
    rate-per-mile x miles driven, and a pooled gradient-boosting model trained on all states at once
    (its inputs include miles-driven growth and miles driven per person).
- State miles driven: FHWA Highway Statistics VM-202, 1995-2024. For 2025-2029 each state keeps
  its 2024 share of national miles driven, applied to the national miles-driven path used by the
  national model (2025 actual, 2026 year-to-date, then trend growth).
- As in the national model, the backtest feeds models the miles actually driven in the years they
  forecast, so it measures skill given miles driven, not the extra error from projecting them.
- Ranking uses weighted absolute percent error across all states (WAPE = sum|error| / sum actual),
  so big states are not drowned out by noise in small ones.
- The top three non-baseline models are blended (inverse-error weights), then every year's state
  forecasts are scaled so they add up to the national blend ("top-down reconciliation").
- 80% / 95% ranges come from each state's own reconciled backtest errors, partly pooled with the
  all-state average because 15 backtest runs per state is a small sample.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import GradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from states import STATES
D = sys.argv[1]; H = 5; Y0, Y1 = 1995, 2024; ORIGINS = range(2005, 2020)

# ---------------- data
fars = pd.read_csv(f"{D}/fars_state_annual.csv")
svmt = pd.read_csv(f"{D}/state_vmt.csv").pivot(index="year", columns="state_fips", values="vmt_millions")
nat_vmt = {h["year"]: h["vmt"] * 1000 for h in json.load(open(f"{D}/results.json"))["history"]}  # millions
def fred(series):
    d = pd.read_csv(f"{D}/raw/states/{series}.csv", parse_dates=["observation_date"])
    d["year"] = d.observation_date.dt.year; d["m"] = d.observation_date.dt.month
    return d.rename(columns={series: "v"})
rows = []
for fips, (ab, name) in STATES.items():
    pop = fred(f"{ab}POP").groupby("year").v.mean() / 1000          # millions
    urm = fred(f"{ab}UR"); ur = urm.groupby("year").v.mean()
    last_pop = pop.index.max()
    g = (pop[last_pop] / pop[last_pop - 5]) ** (1 / 5) - 1          # recent 5-year growth
    for y in range(last_pop + 1, 2030): pop[y] = pop[y - 1] * (1 + g)
    for y in range(int(ur.index.max()) + 1, 2030): ur[y] = ur[ur.index.max()]  # hold latest level
    v = svmt[fips].copy(); share = v[2024] / svmt.loc[2024].sum()
    for y in range(2025, 2030): v[y] = share * nat_vmt[y]
    f = fars[fars.state_fips == fips].set_index("year").fatalities
    for y in range(Y0, 2030):
        rows.append({"fips": fips, "st": ab, "name": name, "year": y,
                     "fat": float(f.get(y, np.nan)) if y <= Y1 else np.nan,
                     "pop": float(pop[y]), "unemp": float(ur[y]), "pop_growth": g, "vmt": float(v[y])})
P = pd.DataFrame(rows)
P["rate"] = P.fat / P["pop"] / 10                                    # deaths per 100k people
P["rate_vmt"] = P.fat / P.vmt * 100                                  # deaths per 100M miles
nat_rate = P.groupby("year").apply(lambda d: d.fat.sum() / d["pop"].sum() / 10)
P["nat_rate"] = P.year.map(nat_rate)

# ---------------- per-state models: f(series up to T, future pop) -> H forecasts
def m_naive(y, pf): return np.repeat(y[-1], H)
def m_avg3(y, pf):  return np.repeat(y[-3:].mean(), H)
def m_holt(y, pf):
    return np.exp(ExponentialSmoothing(np.log(y), trend="add", damped_trend=True).fit().forecast(H))
def m_arima(y, pf):
    best = None
    for p in (0, 1):
        for q in (0, 1):
            try:
                r = ARIMA(np.log(y), order=(p, 1, q)).fit()
                if best is None or r.aic < best.aic: best = r
            except Exception: pass
    return np.exp(best.forecast(H))
def m_rate(y, pf, pop):
    r = np.log(y / pop)
    return np.exp(ExponentialSmoothing(r, trend="add", damped_trend=True).fit().forecast(H)) * pf

def m_rate_vmt(y, vf, vmt):
    r = np.log(y / vmt)
    return np.exp(ExponentialSmoothing(r, trend="add", damped_trend=True).fit().forecast(H)) * vf

# pooled gradient boosting: predicts log(rate[t+h] / 3-yr avg rate[t]) from features at t
def features(df, T):
    """One row per state for origin year T, using only data through T."""
    out = []
    for fips, d in df.groupby("fips"):
        d = d.set_index("year"); r = d.rate
        out.append({"fips": fips, "T": T,
            "lr": np.log(r[T]), "lavg3": np.log(r.loc[T-2:T].mean()),
            "d1": np.log(r[T] / r[T-1]), "d3": np.log(r[T] / r[T-3]),
            "rel": np.log(r[T] / d.nat_rate[T]), "nat_d1": np.log(d.nat_rate[T] / d.nat_rate[T-1]),
            "unemp": d.unemp[T], "dunemp": d.unemp[T] - d.unemp[T-1], "pg": np.log(d["pop"][T] / d["pop"][T-1]),
            "vg": np.log(d.vmt[T] / d.vmt[T-1]), "vpc": np.log(d.vmt[T] / d["pop"][T]), "lrv": np.log(d.rate_vmt[T])})
    return pd.DataFrame(out).set_index("fips")
FEATS = ["lr", "lavg3", "d1", "d3", "rel", "nat_d1", "unemp", "dunemp", "pg", "vg", "vpc", "lrv", "vg_fut"]
FCACHE = {T: features(P, T) for T in range(Y0 + 3, Y1 + 1)}
VM = P.pivot(index="year", columns="fips", values="vmt")
VG = {(t, h): np.log(VM.loc[t + h] / VM.loc[t]).loc[FCACHE[t].index].values   # miles growth t -> t+h
      for t in FCACHE for h in range(1, H + 1) if t + h <= 2029}
def gbm_forecast(T):
    """Train on all origins t with t+h <= T, return {fips: H forecasts of deaths}."""
    preds = {}
    Ft = FCACHE[T]
    for h in range(1, H + 1):
        X, yv = [], []
        for t in range(Y0 + 3, T - h + 1):
            F = FCACHE[t].assign(vg_fut=VG[(t, h)]); tgt = P[P.year == t + h].set_index("fips").rate
            X.append(F[FEATS].values); yv.append(np.log(tgt.loc[F.index].values) - F.lavg3.values)
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                      subsample=0.8, min_samples_leaf=20, random_state=0)
        m.fit(np.vstack(X), np.concatenate(yv))
        Fh = Ft.assign(vg_fut=VG[(T, h)])
        lr_pred = Fh.lavg3.values + m.predict(Fh[FEATS].values)
        for fips, v in zip(Ft.index, lr_pred):
            pf = P[(P.fips == fips) & (P.year == T + h)]["pop"].iloc[0]
            preds.setdefault(fips, []).append(float(np.exp(v) * pf * 10))
    return preds

NAMES = ["Naive (last year)", "3-year average", "Damped trend (Holt)", "ARIMA (auto)",
         "Rate per person x population", "Pooled gradient boosting (all states)", "Rate per mile x miles driven"]
def run_all(T):
    """All model forecasts for origin T: {model: {fips: array(H)}}"""
    res = {k: {} for k in NAMES}
    g = gbm_forecast(T)
    for fips, d in P.groupby("fips"):
        d = d.set_index("year"); y = d.fat.loc[Y0:T].values; pop = d["pop"].loc[Y0:T].values
        pf = d["pop"].loc[T + 1:T + H].values
        res[NAMES[0]][fips] = m_naive(y, pf); res[NAMES[1]][fips] = m_avg3(y, pf)
        res[NAMES[2]][fips] = m_holt(y, pf);  res[NAMES[3]][fips] = m_arima(y, pf)
        res[NAMES[4]][fips] = m_rate(y, pf, pop); res[NAMES[5]][fips] = np.array(g[fips])
        res[NAMES[6]][fips] = m_rate_vmt(y, d.vmt.loc[T + 1:T + H].values, d.vmt.loc[Y0:T].values)
    return res

# ---------------- national blend backtest predictions (for reconciliation in the backtest)
nat = json.load(open(f"{D}/results.json"))
nbt = pd.DataFrame(nat["backtest"]); w_nat = dict(zip(nat["top"], nat["weights"]))
nbt = nbt[nbt.model.isin(nat["top"])].assign(wp=lambda d: d.pred * d.model.map(w_nat))
NAT_BT = nbt.groupby(["origin", "year"]).wp.sum().to_dict()
NAT_FC = {e["year"]: e["pred"] for e in nat["ensemble"]}

# ---------------- backtest
recs = []
for T in ORIGINS:
    res = run_all(T)
    for k in NAMES:
        for fips, p in res[k].items():
            for h in range(1, H + 1):
                a = P[(P.fips == fips) & (P.year == T + h)].fat.iloc[0]
                recs.append({"model": k, "origin": T, "h": h, "year": T + h, "fips": fips, "pred": float(p[h-1]), "actual": a})
    print("backtest origin", T, "done", flush=True)
bt = pd.DataFrame(recs)
def wape(d): return float((d.pred - d.actual).abs().sum() / d.actual.sum() * 100)
score = {k: {"wape_all": wape(bt[bt.model == k]),
             "wape_by_h": [wape(bt[(bt.model == k) & (bt.h == h)]) for h in range(1, H + 1)]} for k in NAMES}
rank = sorted(NAMES, key=lambda k: score[k]["wape_all"])
top = [k for k in rank if k != NAMES[0]][:3]
w = np.array([1 / score[k]["wape_all"] for k in top]); w /= w.sum(); W = dict(zip(top, w))
for k in rank: print(f"{k:40s} {score[k]['wape_all']:5.2f}%", [round(x, 1) for x in score[k]["wape_by_h"]])
print("blend:", top, np.round(w, 3))

# blended + reconciled backtest
b = bt[bt.model.isin(top)].assign(wp=lambda d: d.pred * d.model.map(W))
eb = b.groupby(["origin", "h", "year", "fips"]).agg(raw=("wp", "sum"), actual=("actual", "first")).reset_index()
eb["scale"] = eb.apply(lambda r: NAT_BT[(r.origin, r.year)], axis=1) / eb.groupby(["origin", "year"]).raw.transform("sum")
eb["pred"] = eb.raw * eb.scale
nb = bt[bt.model == NAMES[0]]
score["Blend, reconciled to national"] = {"wape_all": wape(eb), "wape_by_h": [wape(eb[eb.h == h]) for h in range(1, H + 1)]}
score["Blend, before reconciliation"] = {"wape_all": wape(eb.assign(pred=eb.raw)),
                                         "wape_by_h": [wape(eb[eb.h == h].assign(pred=eb.raw)) for h in range(1, H + 1)]}
print("blend raw", round(score["Blend, before reconciliation"]["wape_all"], 2), "reconciled", round(score["Blend, reconciled to national"]["wape_all"], 2))

# interval widths: RMS log error by state & horizon, 50/50 pooled with the all-state value
eb["le"] = np.log(eb.actual.clip(lower=1) / eb.pred)
pooled = eb.groupby("h").le.apply(lambda e: np.mean(e ** 2))
sig = {(f, h): np.sqrt(0.5 * np.mean(g ** 2) + 0.5 * pooled[h]) for (f, h), g in eb.groupby(["fips", "h"]).le}

# ---------------- final forecasts from 2024
res = run_all(Y1)
raw = {f: sum(W[k] * res[k][f] for k in top) for f in STATES}
tot = np.sum([raw[f] for f in STATES], axis=0)
scale = np.array([NAT_FC[Y1 + h] for h in range(1, H + 1)]) / tot
out_states, csv_rows = [], []
for fips, (ab, name) in STATES.items():
    d = P[P.fips == fips].set_index("year"); fc = raw[fips] * scale
    s_bt = eb[eb.fips == fips]; n_bt = nb[nb.fips == fips]
    ens = []
    for h in range(1, H + 1):
        s = sig[(fips, h)]; y = Y1 + h; pop = d["pop"][y]
        e = {"year": y, "pred": float(fc[h-1]), "lo80": float(fc[h-1] * np.exp(-1.2816 * s)), "hi80": float(fc[h-1] * np.exp(1.2816 * s)),
             "lo95": float(fc[h-1] * np.exp(-1.96 * s)), "hi95": float(fc[h-1] * np.exp(1.96 * s)), "rate": float(fc[h-1] / pop / 10),
            "rate_vmt": float(fc[h-1] / d.vmt[y] * 100)}
        ens.append(e)
        csv_rows.append({"state": name, "abbr": ab, "fips": fips, "year": y, "forecast": round(e["pred"]),
                         "lo80": round(e["lo80"]), "hi80": round(e["hi80"]), "lo95": round(e["lo95"]), "hi95": round(e["hi95"]),
                         "deaths_per_100k": round(e["rate"], 2), "deaths_per_100m_miles": round(e["rate_vmt"], 3),
                         "miles_driven_millions": round(d.vmt[y])})
    out_states.append({"fips": fips, "st": ab, "name": name,
        "hist": [{"year": int(y), "fat": float(d.fat[y]), "rate": float(d.rate[y]), "vmt": float(d.vmt[y]),
                  "rate_vmt": float(d.rate_vmt[y])} for y in range(Y0, Y1 + 1)],
        "vmt_fc": [float(d.vmt[Y1 + h]) for h in range(1, H + 1)],
        "pop2029": float(d["pop"][2029]), "ens": ens,
        "models": {k: [float(v) for v in res[k][fips]] for k in NAMES},
        "wape_blend": wape(s_bt), "wape_naive": wape(n_bt), "wape_blend_h5": wape(s_bt[s_bt.h == 5])})
pd.DataFrame(csv_rows).to_csv(f"{D}/state_forecasts.csv", index=False)
json.dump({"states": out_states, "score": score, "rank": rank, "top": top, "weights": [float(x) for x in w],
           "scale": [float(x) for x in scale], "nat_fc": NAT_FC}, open(f"{D}/state_results.json", "w"), indent=1)
print("reconciliation scale by year:", np.round(scale, 3))
t = pd.DataFrame([{"st": s["st"], "2024": s["hist"][-1]["fat"], "2025": s["ens"][0]["pred"], "2029": s["ens"][4]["pred"],
                   "lo80_29": s["ens"][4]["lo80"], "hi80_29": s["ens"][4]["hi80"], "wape": s["wape_blend"], "naive": s["wape_naive"]} for s in out_states])
print(t.round(0).sort_values("2024", ascending=False).to_string(index=False))
