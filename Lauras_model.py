import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


# Parsing all the data needed
patient_data = pd.read_csv('./breast_cancer_recurrence_classifier/data/pnas_patient_info.csv')
sick_tpm = pd.read_csv('./additional_data/pnas_tpm_96_nodup.txt', sep='\t', header=None)
sick_rc = pd.read_csv('./additional_data/pnas_readcounts_96_nodup.txt', sep='\t', header=None)
healthy_tpm = pd.read_csv('./additional_data/pnas_normal_tpm.txt', sep='\t')
healthy_rc = pd.read_csv('./additional_data/pnas_normal_readcounts.txt', sep='\t')

sick_tpm = sick_tpm.set_index(0)

# add 1 to the tpm and then take the log2 of each value
# required ot add 1 because you cannot take the log of 0
sick_log2 = sick_tpm.apply(lambda x: np.log2(x + 1))
healthy_log2 = healthy_tpm.apply(lambda x: np.log2(x + 1))


def plot_histograms(df):
    """
    Generates histograms for each column of a Pandas DataFrame and plots them
    on the same page.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    num_cols = len(df.columns)
    num_rows = (num_cols + 2) // 3  # Adjust for layout
    fig, axes = plt.subplots(num_rows, 6, figsize=(15, 3 * num_rows))
    axes = axes.flatten()  # Flatten axes array for easier indexing

    for i, column in enumerate(df.columns):
        ax = axes[i]
        df[column].hist(ax=ax, bins=20)
        ax.set_title(column)
        ax.set_yscale('log') # change the y axis to log scale so that it looks more like the paper

    # Remove any unused subplots
    # TODO: check this logic
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_histograms(sick_log2)  # plots the tpm then the count of genes that have that tpm for each sample in the sick data set
plot_histograms(healthy_log2)  # does the same for each patient in the healthy data set

# dropped data that was unlabeled
sick_rc_clean = sick_rc[0:60675]
sick_rc_clean = sick_rc_clean.set_index(0)
healthy_rc_clean = healthy_rc[0:60675]
print(sick_rc_clean)

# find the count for number of transcripts sequenced in the data set
sick_rc_clean['sick count'] = sick_rc_clean.sum(axis=1)
healthy_rc_clean['healthy count'] = healthy_rc_clean.sum(axis=1)

print(healthy_rc_clean)

# concatenate so that we have the total read counts for each gene in the sick and healthy populatons, respectively
total_counts = pd.concat([sick_rc_clean['sick count'], healthy_rc_clean['healthy count']], axis=1)
print(total_counts)

# calculate log2 fold change
# is it bad to do log fold change on counts instead of tpm? I do not think so, because we are comparing each gene to itself

total_counts['sick count ratio'] = (total_counts['sick count'] + 1 )/ total_counts['sick count'].sum()
total_counts['healthy count ratio'] = (total_counts['healthy count'] + 1) / total_counts['healthy count'].sum()
total_counts['log sick'] =np.log2(total_counts['sick count ratio'])
total_counts['log healthy'] =np.log2(total_counts['healthy count ratio'])
total_counts['fold change'] = total_counts['log sick'] - total_counts['log healthy']
print(total_counts)

# sort data by fold change
# positive fodl change means the gene is enriched in sick pateints, negative means it is depleted
total_counts = total_counts.sort_values(by='fold change', ascending=False)
print(total_counts)

# get absolute value of the fold change
total_counts['abs value'] = total_counts['fold change'].abs()
print(total_counts)

# select 400 genes with the highest abs(fold change)
# can change this to be a different subset of genes, perhaps only negative/positive fold change, or a different number of genes
max_genes = total_counts.nlargest(10000, 'abs value')
print(max_genes)

# need to explore the max_gene

# histogram of fold changes-- we can porbably assume this is normal
total_counts['fold change'].hist(bins = 40) #histogram of FDR values

# Transpose the subset of data such that the columns are genes and each row is a patient
# add a row that says if the patient is sick or healthy
# 1 means sick, 0 means healthy
sick_tpm_minimized = sick_tpm[sick_tpm.index.isin(max_genes.index.to_list())].T
sick_tpm_minimized['sick'] = 1
healthy_tpm_minimized = healthy_tpm[healthy_tpm.index.isin(max_genes.index.to_list())].T
healthy_tpm_minimized['sick'] = 0

print(sick_tpm_minimized)

# concatenate the data from the sick and healthy patients
clean_data = pd.concat([sick_tpm_minimized, healthy_tpm_minimized])
print(clean_data)

a = clean_data.groupby('sick').agg(['mean', 'std'])
print(a)

# Step 1: Transpose so each gene is a row, and you can compare easily
a_t = a.T  # shape: (genes x 2-level index of sick)

# Step 2: Separate mean and std for row 0 and row 1
mean_0 = a_t.xs('mean', level=1)[0]  # mean values from row 0 (first group, e.g., sick=0)
std_0 = a_t.xs('std', level=1)[0]    # std from row 0
mean_1 = a_t.xs('mean', level=1)[1]  # mean values from row 1 (second group, e.g., sick=1)

# Step 3: Check if mean_1 is within one std of mean_0
within_range = (mean_1 >= mean_0 - std_0) & (mean_1 <= mean_0 + std_0)

# Step 4: Keep only the genes where mean_1 is NOT within one std of mean_0
genes_to_keep = within_range[~within_range].index

# also want to find the count of samples that have a tpm of 0 for the gene of interest
# remove any genes here where the means of the sick and healthy sample are within one stdev of eachother
genes_to_keep = genes_to_keep.to_list()
genes_to_keep.append('sick')
clean_data = clean_data[genes_to_keep]
print(clean_data)

health = clean_data[clean_data['sick'] == 0]
sick = clean_data[clean_data['sick'] == 1]
zero_counts_healthy = (health == 0).sum(axis=0)/32
zero_counts_sick = (sick == 0).sum(axis=0)/96
sum_zero = pd.DataFrame({'healthy': zero_counts_healthy, 'sick': zero_counts_sick})
sum_zero = sum_zero.drop('sick')
print(sum_zero)

# find the proportion of patients in each category that have a TPM = 0 for each gene

# filter out the genes that have TPM = 0 for at least 50% of patients in each group
bad_genes = sum_zero[(sum_zero['healthy'] > 0.5) & (sum_zero['sick'] > 0.5)]
print(bad_genes)
# genes to drop, most of these are pseudoened
drop_genes = bad_genes.index.to_list()

clean_data = clean_data.drop(drop_genes, axis=1)
print(clean_data)
# filtered down the genes to only the essential ones to include

print(clean_data.columns)  # should define what these genes we consider essential are
# worrying about overfitting since we have such few genes...


def run_PCA(data, class_col, plot_label):
    X = data.drop(class_col, axis=1)
    Y = data[class_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2) # change this for more components
    X_pca = pca.fit_transform(X_scaled)
    print(X_pca[:2])
    print("Explained variance:", pca.explained_variance_ratio_)
    print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

    # Figures comparing data before and after PCA
    plt.figure(figsize=(8,6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=Y, cmap='coolwarm', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Data (First Two Features)")
    plt.colorbar(label=plot_label)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='coolwarm', edgecolor='k')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Transformed Data")
    plt.colorbar(label=plot_label)
    plt.show()

    return X_pca, Y

# # x values are each of the genes
# # y values is their health status
# X = clean_data.drop('sick', axis=1)
# y = clean_data['sick']
#
# # import everything needed to complete PCA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # print(X_scaled[:2])
#
# pca = PCA(n_components=2) # change this for more components
# X_pca = pca.fit_transform(X_scaled)
# print(X_pca[:2])
#
#
# print("Explained variance:", pca.explained_variance_ratio_)
# print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))
# #not sure this is a particilar good set of data, but this is how you implement pca
# #pc1 explains 11$ of data, pc2 explains 5%
# #together they explain ~17%
#
# #how data looks before undergoing PCA
#
# plt.figure(figsize=(8,6))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolor='k')
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Original Data (First Two Features)")
# plt.colorbar(label="Sick")
# plt.show()
#
# #how data clusters post PCA
#
# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA Transformed Data")
# plt.colorbar(label="Sick")
# plt.show()

X_pca, y = run_PCA(clean_data, 'sick', "Sick")
#train the model
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5, random_state=42)
#can change the test size, usually should be close to 0.2

#model = LogisticRegression()
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

#
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
#from here we can classify what we precidected for each test and check our accuracy

y_test_array = y_test.to_numpy()
print(y_test_array)
# y_pred

tested = pd.DataFrame({'y_test': y_test_array, 'y_pred': y_pred})
tested['Correct'] = tested['y_test'] == tested['y_pred']
sum(tested['Correct'])/len(tested)
# table to see what y should really be and what was predicted
# returns proportion of correct answers

sum(tested['y_pred'])

# confusion matrix of what each patient's health was determined ot be and what is truly is
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Sick'], # change these labels
            yticklabels=['Healthy', 'Sick']) # change these labels
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_loss = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction Loss:{reconstruction_loss}")

# Predict probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# model is too good, overfitting?

# train the model
# could probably do this in a more efficient way.. not sure how

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5, random_state=42)
# can change the test size, usually should be close to 0.2

models = {"Logistic Regression": LogisticRegression(),
          "Logistic Regression Balanced": LogisticRegression(solver='liblinear', class_weight='balanced'),
          "Logistic Regression liblinear": LogisticRegression(penalty='l1', solver='liblinear'),  # sparse features
          "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
          "SVC": SVC(kernel='linear', probability=True),
          "KNN": KNeighborsClassifier(n_neighbors=5),
          "Logistic Regression Standard Scaler": make_pipeline(StandardScaler(), LogisticRegression())}


def evaluate_model(model, training_data, testing_data, model_name):
    x_training, y_training = training_data
    x_testing, y_testing = testing_data
    model.fit(x_training, y_training)
    y_predict = model.predict(x_testing)

    con_matrix = confusion_matrix(y_testing, y_predict)
    sns.heatmap(con_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Sick'],  # change these labels
                yticklabels=['Healthy', 'Sick'])  # change these labels
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    y_prob = model.predict_proba(x_testing)[:, 1]
    fpr, tpr, threshold = roc_curve(y_testing, y_prob)
    roc_auc = roc_auc_score(y_testing, y_prob)

    return fpr, tpr, roc_auc


roc_auc_stats = []
for m_name, model in models.items():
    curr_roc_auc = evaluate_model(model, (X_train, y_train), (X_test, y_test), m_name)
    roc_auc_stats.append((m_name, curr_roc_auc))

plt.figure(figsize=(6, 6))
colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'purple', 'pink']
for i in range(len(roc_auc_stats)):
    model_name, curr_fpr, curr_tpr, curr_roc_auc = roc_auc_stats[i]
    plt.plot(curr_fpr, curr_tpr, color=colors[i % len(colors)], lw=2, label =f'{model_name} (AUC = {curr_roc_auc:.2f}')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# model1 = LogisticRegression()
# model2 = LogisticRegression(solver='liblinear', class_weight='balanced')
# model3 = LogisticRegression(penalty='l1', solver='liblinear')  # sparse features
# model4 = RandomForestClassifier(n_estimators=100, random_state=42)
# model5 = SVC(kernel='linear', probability=True)
# model6 = KNeighborsClassifier(n_neighbors=5)
# model7 = make_pipeline(StandardScaler(), LogisticRegression())
#
#
#
# model1.fit(X_train, y_train)
# model2.fit(X_train, y_train)
# model3.fit(X_train, y_train)
# model4.fit(X_train, y_train)
# model5.fit(X_train, y_train)
# model6.fit(X_train, y_train)
# model7.fit(X_train, y_train)
#
#
# y_pred1 = model1.predict(X_test)
# y_pred2 = model2.predict(X_test)
# y_pred3 = model3.predict(X_test)
# y_pred4 = model4.predict(X_test)
# y_pred5 = model5.predict(X_test)
# y_pred6 = model6.predict(X_test)
# y_pred7 = model7.predict(X_test)
#
# cm = confusion_matrix(y_test, y_pred1)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-LR')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred2)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-LR Balanced')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred3)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-LR liblinear')
# plt.show()
#
#
# cm = confusion_matrix(y_test, y_pred4)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix- Random Forest')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred5)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix--SVC')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred6)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix- KNN')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred7)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Sick'], #change these labels
#             yticklabels=['Healthy', 'Sick']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix- LR Standard Scaler')
# plt.show()

# # confusion matrix for each model
#
# y_prob1 = model1.predict_proba(X_test)[:, 1]
# y_prob2 = model2.predict_proba(X_test)[:, 1]
# y_prob3 = model3.predict_proba(X_test)[:, 1]
# y_prob4 = model4.predict_proba(X_test)[:, 1]
# y_prob5 = model5.predict_proba(X_test)[:, 1]
# y_prob6 = model6.predict_proba(X_test)[:, 1]
# y_prob7 = model7.predict_proba(X_test)[:, 1]
#
#
# # Compute ROC curve and AUC
# fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob1)
# roc_auc1 = roc_auc_score(y_test, y_prob1)
#
# fpr2, tpr2, thresholds2 = roc_curve(y_test, y_prob2)
# roc_auc2 = roc_auc_score(y_test, y_prob2)
#
# fpr3, tpr3, thresholds3 = roc_curve(y_test, y_prob3)
# roc_auc3 = roc_auc_score(y_test, y_prob3)
#
# fpr4, tpr4, thresholds4 = roc_curve(y_test, y_prob4)
# roc_auc4 = roc_auc_score(y_test, y_prob4)
#
# fpr5, tpr5, thresholds5 = roc_curve(y_test, y_prob5)
# roc_auc5 = roc_auc_score(y_test, y_prob5)
#
# fpr6, tpr6, thresholds6 = roc_curve(y_test, y_prob6)
# roc_auc6 = roc_auc_score(y_test, y_prob6)
#
# fpr7, tpr7, thresholds7 = roc_curve(y_test, y_prob7)
# roc_auc7 = roc_auc_score(y_test, y_prob7)
#
# # Plot the ROC curve
# plt.figure(figsize=(6, 6))
#
# plt.plot(fpr1, tpr1, color='red', lw=2, label='LR (AUC = %0.2f)' % roc_auc1)
# plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='LR-balanced (AUC = %0.2f)' % roc_auc2)
# plt.plot(fpr3, tpr3, color='yellow', lw=2, label='LR-penalty (AUC = %0.2f)' % roc_auc3)
# plt.plot(fpr4, tpr4, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc4)
# plt.plot(fpr5, tpr5, color='blue', lw=2, label='SVC (AUC = %0.2f)' % roc_auc5)
# plt.plot(fpr6, tpr6, color='purple', lw=2, label='KNN (AUC = %0.2f)' % roc_auc6)
# plt.plot(fpr7, tpr7, color='pink', lw=2, label='LR-Scalar (AUC = %0.2f)' % roc_auc7)
#
#
#
#
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# still does not seem right, why is everything a perfect model?

print(patient_data.columns) # how to handle patients with multiple follow ups?

patient_data['cancertype'].unique()

patient_data_clean = patient_data.drop(['Final.Library.Conc..nM.', 'Final.Library.Conc..ng.ul.', 'sample.name.in.MiniSeq',
                                        'sample_id', 'On.Plate', 'readcount', 'avglength', 'fu', 'datespecimen'], axis=1)
print(patient_data_clean) # remove unnecessary columns-- are explaining prepping sample

# change dates to days
patient_data_clean['chemo_duration'] = pd.to_datetime(patient_data_clean['datechemoend'], dayfirst=True) - pd.to_datetime(patient_data_clean['datechemostart'], dayfirst=True)
patient_data_clean['chemo_duration'] = patient_data_clean['chemo_duration'].dt.days
patient_data_clean['since_chemo'] = (pd.to_datetime('01/03/2025') - pd.to_datetime(patient_data_clean['datechemoend'], dayfirst=True)).dt.days
patient_data_clean['time_to_recur'] = (pd.to_datetime(patient_data_clean['daterecurrence'], dayfirst=True) - pd.to_datetime(patient_data_clean['datechemoend'], dayfirst=True)).dt.days
patient_data_clean['time_to_recur'] = np.where(
    patient_data_clean['recurStatus'] == 'N',
    0,
    patient_data_clean['time_to_recur']
)
patient_data_clean = patient_data_clean.drop(['datechemostart', 'datechemoend', 'daterecurrence'], axis=1) #remove the dates
print(patient_data_clean)

# One-hot encode categorical columns
patient_data_clean = pd.get_dummies(patient_data_clean, columns=['cancertype', 'chemo'], dtype=int)
print(patient_data_clean)

patient_data_clean['recurStatus'] = np.where(
    patient_data_clean['recurStatus'] == 'N',
    0,
    1
)
# 1 if their cancer recurrs, 0 otherwise
print(patient_data_clean)

# change percent to a number
patient_data_clean['uniquely_mapped_Percent'] = patient_data_clean['uniquely_mapped_Percent'].str.rstrip('%').astype('float')
print(patient_data_clean)

patient_data_clean[patient_data_clean['recurStatus']==1]['uniquely_mapped_reads'].hist(density=True)
patient_data_clean[patient_data_clean['recurStatus']==0]['uniquely_mapped_reads'].hist(density=True, alpha=0.5)
plt.title('Uniquely Mapped Reads')

patient_data_clean[patient_data_clean['recurStatus']==1]['uniquely_mapped_Percent'].hist(density=True)
patient_data_clean[patient_data_clean['recurStatus']==0]['uniquely_mapped_Percent'].hist(density=True, alpha=0.5)
plt.title('Uniquely Mapped %')

patient_data_clean[patient_data_clean['recurStatus']==1]['unmapped_reads'].hist(density=True)
patient_data_clean[patient_data_clean['recurStatus']==0]['unmapped_reads'].hist(density=True, alpha=0.5)
plt.title('Unmapped Reads %')

print(patient_data_clean.columns)
# should make poiseid the index

patient_data_clean = patient_data_clean.set_index('poiseid')
print(patient_data_clean)

plt.plot(patient_data_clean[patient_data_clean['recurStatus'] == 1]['since_chemo'], patient_data_clean[patient_data_clean['recurStatus'] == 1]['time_to_recur'], 'bo')
plt.plot(patient_data_clean[patient_data_clean['recurStatus'] == 0]['since_chemo'], patient_data_clean[patient_data_clean['recurStatus'] == 0]['time_to_recur'], 'r+')
plt.xlabel('Time Since Chemo')
plt.ylabel('Time to Recur')
plt.title('Time Since Chemo vs. Time to Recur')
plt.show()
# you can see that tie since chemo exceeds time to recur for every patient,
# so I do not think that time since chemo or time to recur is going to be useful in this case

patient_data_clean[patient_data_clean['recurStatus'] == 1]['chemo_duration'].hist(density=True)
patient_data_clean[patient_data_clean['recurStatus'] == 0]['chemo_duration'].hist(density=True, alpha=0.5)
plt.title('Chemo Duration')
# unsure chemo duration will be particularly useful...
# should probably normalize these values before doing

print(patient_data_clean.columns)

patient_data_final = patient_data_clean.drop(['chemo_duration', 'since_chemo', 'time_to_recur'], axis=1)  # unuseful columns
print(patient_data_final)

X_pca, y = run_PCA(patient_data_final, 'recurStatus', "Recurred")

# # run PCA first
# X = patient_data_final.drop('recurStatus', axis=1)
# y = patient_data_final['recurStatus']
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# pca = PCA(n_components=2) # change this for more components
# X_pca = pca.fit_transform(X_scaled)
# print(X_pca[:2])
#
#
# print("Explained variance:", pca.explained_variance_ratio_)
# print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))
#
# plt.figure(figsize=(8,6))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolor='k')
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Original Data (First Two Features)")
# plt.colorbar(label="Recurred")
# plt.show()
#
# # how data clusters post PCA
#
# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA Transformed Data")
# plt.colorbar(label="Recurred")
# plt.show()

#clearly did not seperate the data very well, need to do some other type of transformation

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5, random_state=42)
#can change the test size, usually should be close to 0.2

#model = LogisticRegression()
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

#

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

y_test_array = y_test.to_numpy()
print(y_test_array)
#y_pred

tested = pd.DataFrame({'y_test': y_test_array, 'y_pred': y_pred})
tested['Correct'] = tested['y_test'] == tested['y_pred']
sum(tested['Correct'])/len(tested)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Recurred'], #change these labels
            yticklabels=['Healthy', 'Recurred']) #change these labels
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# repeat what I did earlier by using multiple different models

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5, random_state=42)
# can change the test size, usually should be close to 0.2

models = {"Logistic Regression": LogisticRegression(),
          "Logistic Regression Balanced": LogisticRegression(solver='liblinear', class_weight='balanced'),
          "Logistic Regression Penalty": LogisticRegression(penalty='l1', solver='liblinear'),  # sparse features
          "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
          "SVC": SVC(kernel='linear', probability=True),
          "KNN": KNeighborsClassifier(n_neighbors=5),
          "Logistic Regression Scaler": make_pipeline(StandardScaler(), LogisticRegression())}

roc_auc_stats = []
for m_name, model in models.items():
    curr_roc_auc = evaluate_model(model, (X_train, y_train), (), m_name)
    roc_auc_stats.append((m_name, curr_roc_auc))

plt.figure(figsize=(6, 6))
colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'purple', 'pink']
for i in range(len(roc_auc_stats)):
    model_name, curr_fpr, curr_tpr, curr_roc_auc = roc_auc_stats[i]
    plt.plot(curr_fpr, curr_tpr, color=colors[i % len(colors)], lw=2, label =f'{model_name} (AUC = {curr_roc_auc:.2f}')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# model1 = LogisticRegression()
# model2 = LogisticRegression(solver='liblinear', class_weight='balanced')
# model3 = LogisticRegression(penalty='l1', solver='liblinear')  # sparse features
# model4 = RandomForestClassifier(n_estimators=50, random_state=42)
# model5 = SVC(kernel='linear', probability=True)
# model6 = KNeighborsClassifier(n_neighbors=5)
# model7 = make_pipeline(StandardScaler(), LogisticRegression())
#
#
#
# model1.fit(X_train, y_train)
# model2.fit(X_train, y_train)
# model3.fit(X_train, y_train)
# model4.fit(X_train, y_train)
# model5.fit(X_train, y_train)
# model6.fit(X_train, y_train)
# model7.fit(X_train, y_train)
#
#
# #
#
# y_pred1 = model1.predict(X_test)
# y_pred2 = model2.predict(X_test)
# y_pred3 = model3.predict(X_test)
# y_pred4 = model4.predict(X_test)
# y_pred5 = model5.predict(X_test)
# y_pred6 = model6.predict(X_test)
# y_pred7 = model7.predict(X_test)
#
# cm = confusion_matrix(y_test, y_pred1)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'], #change these labels
#             yticklabels=['Healthy', 'Recurred']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix- LR')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred2)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'], #change these labels
#             yticklabels=['Healthy', 'Recurred']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix- LR Balanced')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred3)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'], #change these labels
#             yticklabels=['Healthy', 'Recurred']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix- LR penalty')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred4)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'], #change these labels
#             yticklabels=['Healthy', 'Recurred']) #change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-Randorm FO=orest')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred5)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'],  # change these labels
#             yticklabels=['Healthy', 'Recurred'])  # change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-SVC')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred6)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'],  # change these labels
#             yticklabels=['Healthy', 'Recurred'])  # change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-KNN')
# plt.show()
#
# cm = confusion_matrix(y_test, y_pred7)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Healthy', 'Recurred'],  # change these labels
#             yticklabels=['Healthy', 'Recurred'])  # change these labels
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix-LR Scaler')
# plt.show()
#
# y_prob1 = model1.predict_proba(X_test)[:, 1]
# y_prob2 = model2.predict_proba(X_test)[:, 1]
# y_prob3 = model3.predict_proba(X_test)[:, 1]
# y_prob4 = model4.predict_proba(X_test)[:, 1]
# y_prob5 = model5.predict_proba(X_test)[:, 1]
# y_prob6 = model6.predict_proba(X_test)[:, 1]
# y_prob7 = model7.predict_proba(X_test)[:, 1]
#
#
# # Compute ROC curve and AUC
# fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob1)
# roc_auc1 = roc_auc_score(y_test, y_prob1)
#
# fpr2, tpr2, thresholds2 = roc_curve(y_test, y_prob2)
# roc_auc2 = roc_auc_score(y_test, y_prob2)
#
# fpr3, tpr3, thresholds3 = roc_curve(y_test, y_prob3)
# roc_auc3 = roc_auc_score(y_test, y_prob3)
#
# fpr4, tpr4, thresholds4 = roc_curve(y_test, y_prob4)
# roc_auc4 = roc_auc_score(y_test, y_prob4)
#
# fpr5, tpr5, thresholds5 = roc_curve(y_test, y_prob5)
# roc_auc5 = roc_auc_score(y_test, y_prob5)
#
# fpr6, tpr6, thresholds6 = roc_curve(y_test, y_prob6)
# roc_auc6 = roc_auc_score(y_test, y_prob6)
#
# fpr7, tpr7, thresholds7 = roc_curve(y_test, y_prob7)
# roc_auc7 = roc_auc_score(y_test, y_prob7)
#
# # Plot the ROC curve
# plt.figure(figsize=(6, 6))
#
# plt.plot(fpr1, tpr1, color='red', lw=2, label='Basic LR (AUC = %0.2f)' % roc_auc1)
# plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='Balanced LR (AUC = %0.2f)' % roc_auc2)
# plt.plot(fpr3, tpr3, color='yellow', lw=2, label='Liblinear LR (AUC = %0.2f)' % roc_auc3)
# plt.plot(fpr4, tpr4, color='green', lw=2, label='Random Forest  (AUC = %0.2f)' % roc_auc4)
# plt.plot(fpr5, tpr5, color='blue', lw=2, label='SVC (AUC = %0.2f)' % roc_auc5)
# plt.plot(fpr6, tpr6, color='purple', lw=2, label='KNN (AUC = %0.2f)' % roc_auc6)
# plt.plot(fpr7, tpr7, color='pink', lw=2, label='LR--Scalar (AUC = %0.2f)' % roc_auc7)
#
#
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
# none of these models seem that great... need to select better variables to use
