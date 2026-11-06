"""Download explanatory series from FRED (St. Louis Fed mirror of official data).
Usage: python src/fetch_fred.py data/raw
  TRFVOLUSM227NFWA  vehicle miles traveled, monthly, millions of miles (FHWA)
  POPTHM            US population, monthly, thousands (Census via BEA)
  UNRATE            unemployment rate, monthly, percent (BLS)
"""
import sys, os, subprocess
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for s in ["TRFVOLUSM227NFWA", "POPTHM", "UNRATE"]:
    subprocess.run(["curl", "-sS", "-m", "120", "--retry", "3", "-o", os.path.join(out, f"{s}.csv"),
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"], check=True)
    print("downloaded", s)
