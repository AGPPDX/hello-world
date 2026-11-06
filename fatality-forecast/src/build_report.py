"""Embed national and state model results into the HTML report.
Usage: python src/build_report.py data/results.json data/state_results.json report/template.html report/traffic-fatality-outlook.html
"""
import sys, json
res, sres, tpl, out = sys.argv[1:5]
r = json.load(open(res)); r.pop("backtest", None)
s = json.load(open(sres))
for st in s["states"]: st.pop("models", None)       # per-model detail stays in the JSON file, not the page
html = open(tpl).read().replace("__DATA__", json.dumps(r, separators=(",", ":"))) \
                      .replace("__STATE_DATA__", json.dumps(s, separators=(",", ":")))
open(out, "w").write(html)
print("wrote", out)
