"""Train, backtest and forecast US traffic fatalities 2025-2029.
Usage: python src/model.py data   (reads data/fars_annual.json + data/raw/*.csv, writes data/results.json)
"""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
D = sys.argv[1]
fars = pd.DataFrame(json.load(open(f"{D}/fars_annual.json"))).set_index("year")
def fred(s, how):
    d = pd.read_csv(f"{D}/raw/{s}.csv", parse_dates=["observation_date"])
    d["year"] = d.observation_date.dt.year; d["m"] = d.observation_date.dt.month
    return d.rename(columns={s: "v"})
vmt_m, pop_m, un_m = fred("TRFVOLUSM227NFWA",0), fred("POPTHM",0), fred("UNRATE",0)
vmt = vmt_m.groupby("year").v.sum() / 1000           # billions of miles
pop = pop_m.groupby("year").v.mean() / 1000          # millions
un  = un_m.groupby("year").v.mean()
# 2026 partial year: scale by YTD growth vs same months of 2025
last_m = vmt_m[vmt_m.year==2026].m.max()
ytd = lambda df, y: df[(df.year==y)&(df.m<=last_m)].v
vmt[2026] = vmt[2025] * ytd(vmt_m,2026).sum()/ytd(vmt_m,2025).sum()
pop[2026] = pop[2025] * ytd(pop_m,2026).mean()/ytd(pop_m,2025).mean()
un[2026]  = un_m[un_m.year==2026].v.mean()
# 2027-2029 baseline scenario: VMT grows at its 2015-2019 + 2023-2025 average pace;
# population at its 2021-2026 average pace; unemployment held at 2026 level
g_vmt = np.mean([vmt[y]/vmt[y-1]-1 for y in list(range(2015,2020))+[2023,2024,2025]])
g_pop = (pop[2026]/pop[2021])**(1/5)-1
for y in range(2027, 2030):
    vmt[y]=vmt[y-1]*(1+g_vmt); pop[y]=pop[y-1]*(1+g_pop); un[y]=un[2026]
df = fars.join(pd.DataFrame({"vmt":vmt,"pop":pop,"unemp":un}), how="outer").loc[1995:2029]
df["rate_vmt"] = df.fatalities/df.vmt*100/1000       # per 100M VMT
df["rate_pop"] = df.fatalities/df["pop"]*100/1000    # per 100k people
H = 5
# ---------- models: each takes train df (years<=T) and exog for future years, returns H forecasts
def m_naive(tr, fut): return np.repeat(tr.fatalities.iloc[-1], H)
def m_drift(tr, fut):
    y=tr.fatalities.values; d=(y[-1]-y[0])/(len(y)-1); return y[-1]+d*np.arange(1,H+1)
def m_holt(tr, fut):
    f=ExponentialSmoothing(np.log(tr.fatalities.values),trend="add",damped_trend=True).fit()
    return np.exp(f.forecast(H))
def m_arima(tr, fut):
    y=np.log(tr.fatalities.values); best=None
    for p in range(3):
        for q in range(3):
            try:
                r=ARIMA(y,order=(p,1,q)).fit()
                if best is None or r.aic<best.aic: best=r
            except Exception: pass
    return np.exp(best.forecast(H))
def m_arimax(tr, fut):
    X=np.column_stack([np.log(tr.vmt),tr.unemp]); Xf=np.column_stack([np.log(fut.vmt),fut.unemp])
    r=ARIMA(np.log(tr.fatalities.values),exog=X,order=(1,0,0),trend="ct").fit()
    return np.exp(r.forecast(H,exog=Xf))
def m_rate(tr, fut):   # forecast fatality rate per mile (damped trend), times projected miles
    f=ExponentialSmoothing(np.log(tr.rate_vmt.values),trend="add",damped_trend=True).fit()
    return np.exp(f.forecast(H))*fut.vmt.values*1000/100
def feats(d):
    return np.column_stack([np.log(d.vmt), d.unemp, np.log(d["pop"]), d.index.values-1995])
def m_ridge(tr, fut):
    m=RidgeCV(alphas=np.logspace(-3,3,25)).fit(feats(tr),np.log(tr.fatalities))
    return np.exp(m.predict(feats(fut)))
def m_gbm(tr, fut):    # predicts log rate per mile; trees can't extrapolate, so rate model
    m=GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
    m.fit(feats(tr),np.log(tr.rate_vmt)); return np.exp(m.predict(feats(fut)))*fut.vmt.values*1000/100
MODELS={"Naive (last year)":m_naive,"Linear drift":m_drift,"Damped trend (Holt)":m_holt,
        "ARIMA (auto)":m_arima,"ARIMAX: miles + unemployment":m_arimax,
        "Rate-per-mile x miles driven":m_rate,"Ridge regression":m_ridge,"Gradient boosting":m_gbm}
# ---------- rolling-origin backtest: train on 1995..T, forecast T+1..T+5, origins 2005..2019
# (exogenous inputs use actual values -> measures model skill given known miles/unemployment)
errs={k:{h:[] for h in range(1,H+1)} for k in MODELS}; bt_rows=[]
for T in range(2005, 2020):
    tr=df.loc[1995:T]; fut=df.loc[T+1:T+H]
    for k,f in MODELS.items():
        p=f(tr,fut)
        for h in range(1,H+1):
            if T+h<=2024:
                a=df.fatalities[T+h]; errs[k][h].append(abs(p[h-1]-a)/a*100)
                bt_rows.append({"model":k,"origin":T,"h":h,"year":T+h,"pred":float(p[h-1]),"actual":float(a)})
bt=pd.DataFrame(bt_rows)
score={k:{"mape_by_h":[float(np.mean(errs[k][h])) for h in range(1,H+1)],
          "mape_all":float(np.mean(sum(errs[k].values(),[])))} for k in MODELS}
rank=sorted(score,key=lambda k:score[k]["mape_all"])
for k in rank: print(f"{k:32s} {score[k]['mape_all']:5.2f}%  by horizon:",[round(x,1) for x in score[k]['mape_by_h']])
# ---------- final fit on 1995-2024, forecast 2025-2029
tr=df.loc[1995:2024]; fut=df.loc[2025:2029]
fc={k:[float(x) for x in MODELS[k](tr,fut)] for k in MODELS}
top=[k for k in rank if k!="Naive (last year)"][:3]
w=np.array([1/score[k]["mape_all"] for k in top]); w/=w.sum()
ens=sum(wi*np.array(fc[k]) for wi,k in zip(w,top))
# ensemble backtest error & empirical 80% / 95% intervals by horizon
ebt=bt[bt.model.isin(top)].copy(); ebt["w"]=ebt.model.map(dict(zip(top,w)))
eb=ebt.assign(wp=ebt.pred*ebt.w).groupby(["origin","h","year"]).agg(pred=("wp","sum"),actual=("actual","first")).reset_index()
eb["logerr"]=np.log(eb.actual/eb.pred)
ens_mape=[float((abs(eb[eb.h==h].actual-eb[eb.h==h].pred)/eb[eb.h==h].actual).mean()*100) for h in range(1,H+1)]
iv={}
for h in range(1,H+1):
    e=eb[eb.h==h].logerr.values; s=np.sqrt(np.mean(e**2))   # RMS log error -> normal-approx band
    iv[h]=(s*1.2816, s*1.96)
ensemble=[{"year":2024+h,"pred":float(ens[h-1]),
  "lo80":float(ens[h-1]*np.exp(-iv[h][0])),"hi80":float(ens[h-1]*np.exp(iv[h][0])),
  "lo95":float(ens[h-1]*np.exp(-iv[h][1])),"hi95":float(ens[h-1]*np.exp(iv[h][1]))} for h in range(1,H+1)]
print("\nTop-3 ensemble:",top,np.round(w,3)," ensemble backtest MAPE by h:",np.round(ens_mape,2))
for r in ensemble: print(r)
for k in rank: print(f"{k:32s}",[round(x) for x in fc[k]])
hist=df.reset_index().rename(columns={"index":"year"})
json.dump({"history":json.loads(hist.to_json(orient="records")),"score":score,"rank":rank,"top":top,
  "weights":[float(x) for x in w],"forecasts":fc,"ensemble":ensemble,"ens_mape":ens_mape,
  "assump":{"g_vmt":g_vmt,"g_pop":g_pop,"unemp":float(un[2026]),"vmt_months_2026":int(last_m)},
  "backtest":json.loads(bt.to_json(orient="records"))},open(f"{D}/results.json","w"),indent=1)
print(df.loc[2015:2029].round(3).to_string())
