import pandas as pd
pd.set_option("display.width", 200)
d = pd.read_csv("results/eatd/eatd_comparison.csv")
c = ["role", "aggregation", "modality", "model", "macro_f1_mean", "dep_f1_mean", "accuracy_mean", "auc_roc_mean", "auc_roc_std"]
d = d[c].round(3)
d.columns = ["role", "aggr", "modality", "model", "macroF1", "depF1", "acc", "AUC", "AUC_std"]
print("\nEATD-Corpus results (5 folds x 3 seeds, test folds never used for tuning)\n")
print(d.to_string(index=False))
print("\nMajority baseline = always predict 'not depressed' (accuracy 0.815, AUC 0.5)\n")