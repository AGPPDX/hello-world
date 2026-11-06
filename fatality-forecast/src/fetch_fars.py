"""Download NHTSA FARS national annual files (1995-2024) and total fatalities per year.
Usage: python src/fetch_fars.py <temp_dir_for_zips> data/fars_annual.json
Each zip (~35 MB) is deleted after its accident.csv is summarized.
"""
import sys, os, io, zipfile, csv, subprocess, json
from concurrent.futures import ThreadPoolExecutor
RAW = sys.argv[1]; OUT = sys.argv[2]
def one(y):
    z = os.path.join(RAW, f"fars{y}.zip")
    url = f"https://static.nhtsa.gov/nhtsa/downloads/FARS/{y}/National/FARS{y}NationalCSV.zip"
    subprocess.run(["curl","-sS","-m","600","--retry","3","-o",z,url], check=True)
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith("accident.csv")][0]
        raw = zf.read(name)
    os.remove(z)
    text = raw.decode("latin-1")
    rd = csv.DictReader(io.StringIO(text))
    rd.fieldnames = [f.strip().upper().lstrip("﻿").strip('"') for f in rd.fieldnames]
    crashes = fat = 0
    for r in rd:
        crashes += 1; fat += int(r["FATALS"])
    return {"year": y, "crashes": crashes, "fatalities": fat}
with ThreadPoolExecutor(6) as ex:
    res = sorted(ex.map(one, range(1995, 2025)), key=lambda d: d["year"])
json.dump(res, open(OUT, "w"), indent=1)
for r in res: print(r)
