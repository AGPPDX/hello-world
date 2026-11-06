"""Download NHTSA FARS national annual files (1995-2024) and total fatalities per year,
nationally and by state.
Usage: python src/fetch_fars.py <temp_dir_for_zips> data/fars_annual.json data/fars_state_annual.csv
Each zip (~35 MB) is deleted after its accident.csv is summarized.
STATE in FARS is the state FIPS code (1 = Alabama ... 56 = Wyoming; 11 = DC).
Puerto Rico (43) and Virgin Islands (52) are not part of the US totals and are dropped.
"""
import sys, os, io, zipfile, csv, subprocess, json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
RAW, OUT, OUT_STATE = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(RAW, exist_ok=True)
TERRITORIES = {43, 52}
def one(y):
    z = os.path.join(RAW, f"fars{y}.zip")
    url = f"https://static.nhtsa.gov/nhtsa/downloads/FARS/{y}/National/FARS{y}NationalCSV.zip"
    subprocess.run(["curl","-sS","-m","600","--retry","3","-o",z,url], check=True)
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith("accident.csv")][0]
        raw = zf.read(name)
    os.remove(z)
    rd = csv.DictReader(io.StringIO(raw.decode("latin-1")))
    # header cleanup: drop a UTF-8 byte-order mark (read as latin-1 it appears as three odd characters) and quotes
    rd.fieldnames = [f.replace("\u00ef\u00bb\u00bf", "").strip().strip('"').upper() for f in rd.fieldnames]
    crashes = fat = 0; st_fat = Counter(); st_cr = Counter()
    for r in rd:
        s = int(r["STATE"]); f = int(r["FATALS"])
        if s in TERRITORIES: continue
        crashes += 1; fat += f; st_fat[s] += f; st_cr[s] += 1
    return {"year": y, "crashes": crashes, "fatalities": fat}, \
           [{"year": y, "state_fips": s, "crashes": st_cr[s], "fatalities": st_fat[s]} for s in sorted(st_fat)]
with ThreadPoolExecutor(6) as ex:
    res = sorted(ex.map(one, range(1995, 2025)), key=lambda d: d[0]["year"])
json.dump([r[0] for r in res], open(OUT, "w"), indent=1)
with open(OUT_STATE, "w", newline="") as fh:
    w = csv.DictWriter(fh, ["year", "state_fips", "crashes", "fatalities"]); w.writeheader()
    for r in res: w.writerows(r[1])
for r in res: print(r[0]["year"], r[0]["fatalities"], "states:", len(r[1]))
