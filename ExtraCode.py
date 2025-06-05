# Code for searching BioMart

from pybiomart import Dataset

# Connect to the Ensembl BioMart server and select the human gene dataset
dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')

gene_list = ['EGFR', 'PTEN', 'MYC', 'BRAF'] # add your gene names as strings here

# Query Ensembl gene IDs for a list of gene symbols
results = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])

filtered_results = results[results['Gene name'].isin(gene_list)]
ensemble_ids = list(filtered_results['Gene stable ID']) # not guaranteed to be in the same order as input list

print(filtered_results)
print(ensemble_ids)

import pandas as pd

# If you have a direct CSV link, use it here. Example for UCI dataset:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

# Define column names as per UCI documentation
col_names = [
    'Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    'Normal Nucleoli', 'Mitoses', 'Class'
]

# Load the data
data = pd.read_csv(url, header=None, names=col_names)

# Drop the sample code column (not needed for prediction)
data = data.drop(['Sample code'], axis=1)

# Show the first few rows and info
print(data.head())
print(data.info())

# STEP 1: Upload files
from google.colab import files
uploaded = files.upload()  # Upload both: pnas_tpm_96_nodup.txt and labels_recurrence_96.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# STEP 2: Load data
expression_file = 'pnas_tpm_96_nodup.txt'
labels_file = 'labels_recurrence_96.csv'

df_expr = pd.read_csv(expression_file, sep='\t', index_col=0).T
df_expr.index.name = 'sample_id'

df_labels = pd.read_csv(labels_file)
df_labels = df_labels.set_index('sample_id')

# Merge expression data and labels
df = df_expr.join(df_labels)

# STEP 3: Define target and features
y = df['Recurrence']
X = df.drop('Recurrence', axis=1)

# STEP 4: Feature selection - top 100 genes
selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X, y)

# STEP 5: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# STEP 6: Models
model1 = LogisticRegression(max_iter=1000)  # Basic LR
model2 = LogisticRegression(class_weight='balanced', max_iter=1000)  # Balanced LR
model3 = LogisticRegression(solver='liblinear', max_iter=1000)  # Liblinear LR
model4 = RandomForestClassifier(n_estimators=100, random_state=42)  # RF
model5 = SVC(probability=True, kernel='rbf', random_state=42)  # SVM
model6 = KNeighborsClassifier(n_neighbors=5)  # KNN

# For model7: Scale the data before logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model7 = LogisticRegression(max_iter=1000)

# STEP 7: Train models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)
model7.fit(X_train_scaled, y_train)

# STEP 8: Predict probabilities
y_prob1 = model1.predict_proba(X_test)[:, 1]
y_prob2 = model2.predict_proba(X_test)[:, 1]
y_prob3 = model3.predict_proba(X_test)[:, 1]
y_prob4 = model4.predict_proba(X_test)[:, 1]
y_prob5 = model5.predict_proba(X_test)[:, 1]
y_prob6 = model6.predict_proba(X_test)[:, 1]
y_prob7 = model7.predict_proba(X_test_scaled)[:, 1]

# STEP 9: Calculate AUCs and ROC curves
fpr1, tpr1, _ = roc_curve(y_test, y_prob1)
roc_auc1 = roc_auc_score(y_test, y_prob1)

fpr2, tpr2, _ = roc_curve(y_test, y_prob2)
roc_auc2 = roc_auc_score(y_test, y_prob2)

fpr3, tpr3, _ = roc_curve(y_test, y_prob3)
roc_auc3 = roc_auc_score(y_test, y_prob3)

fpr4, tpr4, _ = roc_curve(y_test, y_prob4)
roc_auc4 = roc_auc_score(y_test, y_prob4)

fpr5, tpr5, _ = roc_curve(y_test, y_prob5)
roc_auc5 = roc_auc_score(y_test, y_prob5)

fpr6, tpr6, _ = roc_curve(y_test, y_prob6)
roc_auc6 = roc_auc_score(y_test, y_prob6)

fpr7, tpr7, _ = roc_curve(y_test, y_prob7)
roc_auc7 = roc_auc_score(y_test, y_prob7)

# STEP 10: Plot ROC curves
plt.figure(figsize=(7, 7))
plt.plot(fpr1, tpr1, color='red', lw=2, label='Basic LR (AUC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='Balanced LR (AUC = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='yellow', lw=2, label='Liblinear LR (AUC = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc4)
plt.plot(fpr5, tpr5, color='blue', lw=2, label='SVM (AUC = %0.2f)' % roc_auc5)
plt.plot(fpr6, tpr6, color='purple', lw=2, label='KNN (AUC = %0.2f)' % roc_auc6)
plt.plot(fpr7, tpr7, color='pink', lw=2, label='Scaled LR (AUC = %0.2f)' % roc_auc7)

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Breast Cancer Recurrence Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()