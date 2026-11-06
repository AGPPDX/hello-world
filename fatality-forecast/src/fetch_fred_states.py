"""Download state population ({ST}POP, annual, thousands; Census) and state unemployment rate
({ST}UR, monthly, percent; BLS) from FRED. Usage: python src/fetch_fred_states.py data/raw/states
"""
import sys, os, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from states import STATES
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def get(series):
    p = os.path.join(out, f"{series}.csv")
    subprocess.run(["curl", "-sS", "-m", "120", "--retry", "3", "-f", "-o", p,
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"], check=True)
    return series
ids = [f"{ab}{s}" for ab, _ in STATES.values() for s in ("POP", "UR")]
with ThreadPoolExecutor(6) as ex:
    for s in ex.map(get, ids): pass
print("downloaded", len(ids), "series")
