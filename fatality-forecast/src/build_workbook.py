"""Build an Excel workbook of the state (and national) forecasts.
Usage: python src/build_workbook.py data data/state_forecasts.xlsx
Reads data/results.json, data/state_results.json, data/fars_annual.json.
Recorded deaths, forecasts, ranges and population are values; totals, changes,
rates and checks are Excel formulas so they recalculate if inputs are edited.
"""
import sys, json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.comments import Comment

D, OUT = sys.argv[1], sys.argv[2]
nat = json.load(open(f"{D}/results.json")); st = json.load(open(f"{D}/state_results.json"))
fars = json.load(open(f"{D}/fars_annual.json"))
S = sorted(st["states"], key=lambda s: s["name"])
YRS = [e["year"] for e in S[0]["ens"]]; HIST = [h["year"] for h in S[0]["hist"]]

F = "Arial"
H_FONT = Font(name=F, bold=True, color="FFFFFF"); H_FILL = PatternFill("solid", fgColor="0F5A3C")
BODY = Font(name=F); BOLD = Font(name=F, bold=True); NOTE = Font(name=F, italic=True, color="555555", size=9)
TITLE = Font(name=F, bold=True, size=14); CALC_FILL = PatternFill("solid", fgColor="EEF4F0")
TOT_FILL = PatternFill("solid", fgColor="DDE8E1"); THIN = Side(style="thin", color="B7C4BC")
INT, PCT, RATE = "#,##0", "0.0%;-0.0%;0.0%", "0.00"
SRC = ("Source: NHTSA Fatality Analysis Reporting System (FARS) national annual files 1995-2024, "
       "static.nhtsa.gov; forecasts from this project's models (see Read Me).")

def header(ws, row, labels, widths=None, calc_cols=()):
    for i, t in enumerate(labels, 1):
        c = ws.cell(row, i, t); c.font = H_FONT; c.fill = H_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 32
    if widths:
        for i, w in enumerate(widths, 1): ws.column_dimensions[L(i)].width = w
def fmt(c, nf=None, font=BODY, fill=None):
    c.font = font
    if nf: c.number_format = nf
    if fill: c.fill = fill
    return c

wb = Workbook()

# ---------------- Read Me
ws = wb.active; ws.title = "Read Me"
ws.column_dimensions["A"].width = 26; ws.column_dimensions["B"].width = 100
rows = [
 ("US Traffic Fatality Forecasts by State, 2025-2029", None),
 (None, None),
 ("What this is", "Forecasts of annual motor-vehicle traffic deaths for the 50 states and DC, 2025-2029, from 30 years (1995-2024) of NHTSA data. State forecasts are scaled so they add up to the national forecast each year."),
 ("Fatality data", "NHTSA Fatality Analysis Reporting System (FARS), national annual files 1995-2024 (static.nhtsa.gov). Deaths = sum of FATALS across crash records. Puerto Rico and the Virgin Islands are excluded, matching NHTSA US totals. 2024 may still be revised by NHTSA; the 2025 file is not yet published."),
 ("Population", "Census state population via FRED ({ST}POP), actuals through 2025. After 2025 each state grows at its own 2020-2025 average rate."),
 ("State models", "Six models per state: naive (last year), 3-year average, damped trend, ARIMA, rate per person x population, and a gradient-boosting model trained on all states together. The three most accurate in backtests (ARIMA, 3-year average, damped trend) are blended with inverse-error weights, then scaled to the national forecast."),
 ("National model", "Eight models including ARIMAX with miles driven and unemployment; the blend of gradient boosting, ARIMA and linear drift is used. See the National sheet."),
 ("Accuracy", f"Backtest: models trained through each year 2005-2019 and scored on the next 1-5 years. State blend error (weighted across states): {st['score']['Blend, reconciled to national']['wape_by_h'][0]:.1f}% one year ahead, {st['score']['Blend, reconciled to national']['wape_by_h'][4]:.1f}% five years ahead - about the same as assuming each state repeats its last year. Small states swing the most."),
 ("Ranges", "80% range: in backtests, actual deaths fell inside a band this wide about 80% of the time. 95% range likewise. Built from each state's own backtest errors, partly pooled with the all-state average."),
 (None, None),
 ("Sheets", None),
 ("Summary", "One row per state: 2024 recorded deaths, 2025-2029 forecasts, 2029 range, change since 2024, 2029 death rate, and backtest error. Totals and a check against the national forecast at the bottom."),
 ("Forecast Detail", "One row per state and year (255 rows) with 80% and 95% ranges, population and death rate. Easy to load into a BI tool or pivot table."),
 ("History", "Recorded deaths by state, 1995-2024."),
 ("National", "National recorded deaths 1995-2024 and the national forecast with ranges."),
 (None, None),
 ("Shaded columns", "Light green shading marks Excel formulas (totals, changes, rates, checks). Other numbers are data or model output."),
 ("Prepared", "November 2026. Model estimates, not official NHTSA projections. Code: fatality-forecast/ in the AGPPDX/hello-world repository."),
]
for r, (a, b) in enumerate(rows, 1):
    if a: ws.cell(r, 1, a).font = TITLE if r == 1 else BOLD
    if b:
        c = ws.cell(r, 2, b); c.font = BODY; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.cell(r, 1).alignment = Alignment(vertical="top")

# ---------------- Summary
ws = wb.create_sheet("Summary")
cols = ["State", "Abbr.", "2024 recorded"] + [f"{y} forecast" for y in YRS] + \
       ["2029 low (80%)", "2029 high (80%)", "Change 2024 to 2029", "2029 population", "2029 deaths per 100k",
        "Backtest error (blend)", "Backtest error (last-year baseline)"]
header(ws, 1, cols, [22, 7, 11, 11, 11, 11, 11, 11, 12, 12, 12, 14, 12, 12, 14])
n = len(S); first, last = 2, 1 + n
for i, s in enumerate(S):
    r = first + i; e5 = s["ens"][-1]
    fmt(ws.cell(r, 1, s["name"])); fmt(ws.cell(r, 2, s["st"]))
    fmt(ws.cell(r, 3, s["hist"][-1]["fat"]), INT)
    for j, e in enumerate(s["ens"]): fmt(ws.cell(r, 4 + j, round(e["pred"])), INT)
    fmt(ws.cell(r, 9, round(e5["lo80"])), INT); fmt(ws.cell(r, 10, round(e5["hi80"])), INT)
    fmt(ws.cell(r, 11, f"=H{r}/C{r}-1"), PCT, fill=CALC_FILL)
    fmt(ws.cell(r, 12, round(e5["pred"] / e5["rate"] * 1e5)), INT)
    fmt(ws.cell(r, 13, f"=IF(L{r}>0,H{r}/L{r}*100000,0)"), RATE, fill=CALC_FILL)
    fmt(ws.cell(r, 14, s["wape_blend"] / 100), PCT); fmt(ws.cell(r, 15, s["wape_naive"] / 100), PCT)
t = last + 1
fmt(ws.cell(t, 1, "Total, 50 states + DC"), font=BOLD, fill=TOT_FILL); ws.cell(t, 2).fill = TOT_FILL
for c in range(3, 16): ws.cell(t, c).fill = TOT_FILL
for c in [3, 4, 5, 6, 7, 8, 12]:
    fmt(ws.cell(t, c, f"=SUM({L(c)}{first}:{L(c)}{last})"), INT, BOLD, TOT_FILL)
fmt(ws.cell(t, 11, f"=H{t}/C{t}-1"), PCT, BOLD, TOT_FILL)
fmt(ws.cell(t, 13, f"=IF(L{t}>0,H{t}/L{t}*100000,0)"), RATE, BOLD, TOT_FILL)
for c in range(1, 16): ws.cell(t, c).border = Border(top=THIN)
ws.cell(t, 9, "Ranges don't add").font = NOTE
nr = t + 1
fmt(ws.cell(nr, 1, "National forecast (National sheet)"))
for j in range(len(YRS)):
    fmt(ws.cell(nr, 4 + j, f"=National!B{len(HIST) + 2 + j}"), INT, fill=CALC_FILL)
fmt(ws.cell(nr, 3, f"=National!B{len(HIST) + 1}"), INT, fill=CALC_FILL)
cr = nr + 1
fmt(ws.cell(cr, 1, "Check: states minus national"))
for c in range(3, 9): fmt(ws.cell(cr, c, f"={L(c)}{t}-{L(c)}{nr}"), INT, fill=CALC_FILL)
ws.cell(cr, 9, "Small differences come from rounding each state to whole deaths.").font = NOTE
ws.cell(cr + 2, 1, SRC).font = NOTE
ws.freeze_panes = "C2"; ws.auto_filter.ref = f"A1:O{last}"
ws["N1"].comment = Comment("Weighted average absolute error of this state's blended forecasts in the 2005-2019 backtest, all horizons 1-5 years.", "Forecast")
ws["O1"].comment = Comment("Same measure for simply repeating the last recorded year. Lower is better.", "Forecast")
ws["L1"].comment = Comment("Census state population via FRED, projected past 2025 at each state's 2020-2025 growth rate.", "Forecast")

# ---------------- Forecast Detail (long format)
ws = wb.create_sheet("Forecast Detail")
header(ws, 1, ["State", "Abbr.", "FIPS", "Year", "Forecast", "Low (80%)", "High (80%)", "Low (95%)", "High (95%)",
               "Population", "Deaths per 100k"], [22, 7, 7, 7, 11, 11, 11, 11, 11, 13, 12])
r = 2
for s in S:
    for e in s["ens"]:
        vals = [s["name"], s["st"], s["fips"], str(e["year"]), round(e["pred"]), round(e["lo80"]), round(e["hi80"]),
                round(e["lo95"]), round(e["hi95"]), round(e["pred"] / e["rate"] * 1e5)]
        for c, v in enumerate(vals, 1): fmt(ws.cell(r, c, v), INT if c >= 5 else None)
        fmt(ws.cell(r, 11, f"=IF(J{r}>0,E{r}/J{r}*100000,0)"), RATE, fill=CALC_FILL)
        r += 1
ws.cell(r + 1, 1, SRC).font = NOTE
ws.freeze_panes = "E2"; ws.auto_filter.ref = f"A1:K{r - 1}"

# ---------------- History (wide)
ws = wb.create_sheet("History")
header(ws, 1, ["State", "Abbr."] + [str(y) for y in HIST], [22, 7] + [8] * len(HIST))
for i, s in enumerate(S):
    r = 2 + i; fmt(ws.cell(r, 1, s["name"])); fmt(ws.cell(r, 2, s["st"]))
    for j, h in enumerate(s["hist"]): fmt(ws.cell(r, 3 + j, h["fat"]), INT)
t = 2 + len(S)
fmt(ws.cell(t, 1, "Total, 50 states + DC"), font=BOLD, fill=TOT_FILL); ws.cell(t, 2).fill = TOT_FILL
for j in range(len(HIST)):
    c = 3 + j; fmt(ws.cell(t, c, f"=SUM({L(c)}2:{L(c)}{t - 1})"), INT, BOLD, TOT_FILL)
ws.cell(t + 2, 1, SRC).font = NOTE
ws.freeze_panes = "C2"

# ---------------- National
ws = wb.create_sheet("National")
header(ws, 1, ["Year", "Deaths (recorded or forecast)", "Type", "Low (80%)", "High (80%)", "Low (95%)", "High (95%)"],
       [8, 16, 11, 12, 12, 12, 12])
r = 2
for f in fars:
    fmt(ws.cell(r, 1, str(f["year"]))); fmt(ws.cell(r, 2, f["fatalities"]), INT); fmt(ws.cell(r, 3, "Recorded")); r += 1
for e in nat["ensemble"]:
    fmt(ws.cell(r, 1, str(e["year"]))); fmt(ws.cell(r, 2, round(e["pred"])), INT); fmt(ws.cell(r, 3, "Forecast"))
    for c, k in zip(range(4, 8), ["lo80", "hi80", "lo95", "hi95"]): fmt(ws.cell(r, c, round(e[k])), INT)
    r += 1
fmt(ws.cell(r + 1, 1, "Total forecast deaths, 2025-2029"), font=BOLD)
fmt(ws.cell(r + 1, 2, f"=SUM(B{r - len(YRS)}:B{r - 1})"), INT, BOLD, CALC_FILL)
ws.cell(r + 3, 1, SRC).font = NOTE
ws.freeze_panes = "A2"

wb.save(OUT); print("wrote", OUT)
