"""Embed model results into the HTML report.
Usage: python src/build_report.py data/results.json report/template.html report/traffic-fatality-outlook.html
"""
import sys, json
res, tpl, out = sys.argv[1:4]
r = json.load(open(res)); r.pop("backtest", None)
open(out, "w").write(open(tpl).read().replace("__DATA__", json.dumps(r, separators=(",", ":"))))
print("wrote", out)
