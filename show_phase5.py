import glob
import pandas as pd
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
out = ""
files = sorted(glob.glob("results/daic*.csv") + glob.glob("results/daic/*.csv"))
for f in files:
    d = pd.read_csv(f)
    n = 8 if "predictions" in f else 40
    out += "\n===== " + f + " " + str(d.shape) + "\n"
    out += str(d.columns.tolist()) + "\n"
    out += d.head(n).round(3).to_string(index=False) + "\n"
open("results/summary_phase5.txt", "w").write(out)
print("done")