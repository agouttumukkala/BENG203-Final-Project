# Converted sample code to Python

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load data
tpm = pd.read_csv("data/tpm_96_nodup.tsv", sep="\t")
readcounts = pd.read_csv("data/readcounts_96_nodup.tsv", sep="\t")
patient_info = pd.read_csv("data/patient_info.csv")

# Match sample order
sample_ids = readcounts.columns[1:]
patient_info = patient_info.set_index("sampleID").loc[sample_ids].reset_index()
sampel_recurStatus = patient_info[["sampleID", "recurStatus"]]

# Load gene list
with open("data/preselectedList", "r") as f:
    preselectedList = [line.strip() for line in f.readlines()]

# Filter for marker genes
readcounts_filtered = readcounts[readcounts["gene_id"].isin(preselectedList)].set_index("gene_id")

# Use variance as a proxy for DE significance
gene_variances = readcounts_filtered.var(axis=1)
rank_marker_gene = gene_variances.sort_values(ascending=False).index.tolist()

# Prep TPM data
tpm = tpm.set_index("gene_id")

# Evaluate AUCs at increasing gene set sizes
geneNum = list(range(30, 751, 30))
averageAUC = []

for num in tqdm(geneNum):
    selected_genes = rank_marker_gene[:num]
    temp_tpm = tpm.loc[selected_genes].T
    temp_tpm["status"] = (sampel_recurStatus["recurStatus"] == "R").astype(int)

    idx_pos = temp_tpm[temp_tpm["status"] == 1].index
    idx_neg = temp_tpm[temp_tpm["status"] == 0].index

    aucs = []
    np.random.seed(123)
    for _ in range(100):
        test_idx_pos = np.random.choice(idx_pos, 8, replace=False)
        test_idx_neg = np.random.choice(idx_neg, 48, replace=False)
        test_idx = np.concatenate([test_idx_pos, test_idx_neg])
        train_idx = [i for i in temp_tpm.index if i not in test_idx]

        train = temp_tpm.loc[train_idx].copy()
        test = temp_tpm.loc[test_idx].copy()

        zero_var_cols = (train.drop(columns="status").sum(axis=0) == 0)
        if zero_var_cols.any():
            train = train.loc[:, ~zero_var_cols]
            test = test.loc[:, train.columns]

        X_train, y_train = train.drop(columns="status"), train["status"]
        X_test, y_test = test.drop(columns="status"), test["status"]

        model = SVC(kernel="sigmoid", C=5, probability=True)
        model.fit(X_train, y_train)
        probs = model.decision_function(X_test)
        auc = roc_auc_score(y_test, probs)
        aucs.append(auc)

    averageAUC.append(np.mean(aucs))

# Plot Fig 5C
plt.figure(figsize=(4, 4))
sns.scatterplot(x=geneNum, y=averageAUC, color="red", s=30)
plt.ylim(0, 1)
plt.xlabel("Number of genes")
plt.ylabel("Average AUC")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("averageAUC.pdf")

# Fig 5D with top 215 genes
selected_genes = rank_marker_gene[:215]
temp_tpm = tpm.loc[selected_genes].T
temp_tpm["status"] = (sampel_recurStatus["recurStatus"] == "R").astype(int)

idx_pos = temp_tpm[temp_tpm["status"] == 1].index
idx_neg = temp_tpm[temp_tpm["status"] == 0].index

roc_data = []
auc_values = []

for i in range(3):
    test_idx_pos = np.random.choice(idx_pos, 8, replace=False)
    test_idx_neg = np.random.choice(idx_neg, 48, replace=False)
    test_idx = np.concatenate([test_idx_pos, test_idx_neg])
    train_idx = [i for i in temp_tpm.index if i not in test_idx]

    train = temp_tpm.loc[train_idx].copy()
    test = temp_tpm.loc[test_idx].copy()

    zero_var_cols = (train.drop(columns="status").sum(axis=0) == 0)
    if zero_var_cols.any():
        train = train.loc[:, ~zero_var_cols]
        test = test.loc[:, train.columns]

    X_train, y_train = train.drop(columns="status"), train["status"]
    X_test, y_test = test.drop(columns="status"), test["status"]

    model = SVC(kernel="sigmoid", C=5, probability=True)
    model.fit(X_train, y_train)
    probs = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    auc_values.append(auc)

    roc_data.extend(pd.DataFrame({"x": fpr, "y": tpr, "runtime": i + 1}).to_dict("records"))

# Plot Fig 5D
roc_df = pd.DataFrame(roc_data)
plt.figure(figsize=(4, 4))
sns.lineplot(data=roc_df, x="x", y="y", hue="runtime", palette="tab10")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.annotate(f"Average AUC = {np.mean(auc_values):.3f}", xy=(0.65, 0.2), fontsize=10)
plt.legend().remove()
plt.tight_layout()
plt.savefig("ROCplot.pdf")