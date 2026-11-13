import pandas as pd
d = pd.read_csv("results/eatd/eatd_comparison.csv")
out = d.round(3).to_string()
open("results/summary_comparison.txt", "w").write(out)
print(out)