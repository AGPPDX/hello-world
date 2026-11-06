"""Download FHWA Highway Statistics table VM-202 (annual vehicle-miles of travel by state,
one sheet per year, VM-2 layout) and extract each state's total VMT, 1995-2024.

Usage: python src/fetch_fhwa_vmt.py <temp_dir> data/state_vmt.csv
Source: https://www.fhwa.dot.gov/policyinformation/statistics/2024/xls/vm202.xlsx (November 2025 edition)

Column positions differ between years, so in each sheet the grand-total column is the one
holding the largest value on the "U.S. Total" row (rural + urban combined). Every year is
checked: the 50 states + DC must add up to that U.S. total.
"""
import sys, os, re, csv, subprocess
from openpyxl import load_workbook
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from states import STATES
TMP, OUT = sys.argv[1], sys.argv[2]; os.makedirs(TMP, exist_ok=True)
URL = "https://www.fhwa.dot.gov/policyinformation/statistics/2024/xls/vm202.xlsx"
x = os.path.join(TMP, "vm202.xlsx")
subprocess.run(["curl", "-sS", "-m", "300", "--retry", "3", "-f", "-o", x, URL], check=True)
NAME2FIPS = {n.lower(): f for f, (_, n) in STATES.items()}
def clean(s):
    s = re.sub(r"\(.*?\)|\d+/|\*", "", str(s))           # drop footnote markers like (2), 1/, *
    return re.sub(r"\s+", " ", s).strip().lower().replace("dist. of columbia", "district of columbia")
wb = load_workbook(x, read_only=True, data_only=True)
rows = []
for y in range(1995, 2025):
    ws = wb[str(y)]; data = list(ws.iter_rows(values_only=True))
    tot_row = next(r for r in data if r and r[0] and clean(r[0]) == "u.s. total")
    num = [(j, v) for j, v in enumerate(tot_row[:25]) if isinstance(v, (int, float))]
    col, us_total = max(num, key=lambda t: t[1])
    found = {}
    for r in data:
        if r and r[0] and clean(r[0]) in NAME2FIPS and isinstance(r[col], (int, float)):
            found[NAME2FIPS[clean(r[0])]] = float(r[col])
    missing = set(STATES) - set(found)
    assert not missing, (y, [STATES[m][1] for m in missing])
    s = sum(found.values())
    assert abs(s / us_total - 1) < 0.002, (y, s, us_total)
    rows += [{"year": y, "state_fips": f, "vmt_millions": round(v, 3)} for f, v in sorted(found.items())]
    print(y, f"states sum {s:,.0f} vs U.S. total {us_total:,.0f} million miles (column {col})")
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, ["year", "state_fips", "vmt_millions"]); w.writeheader(); w.writerows(rows)
print("wrote", OUT, len(rows), "rows")
