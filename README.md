import kagglehub

# Download latest version
path = kagglehub.dataset_download("prakharrathi25/banking-dataset-marketing-targets")

print("Path to dataset files:", path)





import pandas as pd
import os

# Construct the full path to the dataset file, using 'train.csv'
data_file_path = os.path.join(path, "train.csv")

# Load the dataset, specifying semicolon as the delimiter
df = pd.read_csv(data_file_path, sep=';')

# Print first 5 rows
print("First 5 Rows:")
print(df.head())

# Print column data types
print("\nColumn Data Types:")
print(df.dtypes)

# Print dataset shape
print("\nDataset Shape (Rows, Columns):")
print(df.shape)








import kagglehub
import pandas as pd
import os

# Load raw data
path = kagglehub.dataset_download("prakharrathi25/banking-dataset-marketing-targets")
data_file_path = os.path.join(path, "train.csv")
df = pd.read_csv(data_file_path, sep=';')

print("Raw shape:", df.shape)

# 1. Drop duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# 2. Drop 'duration' — only known AFTER a call ends, near-perfect proxy for y (target leakage)
df = df.drop(columns=['duration'])

# 3. 'unknown' values kept as their own category (genuine response, not missing data)

df['y'] = df['y'].str.strip().str.lower()

print("Cleaned shape:", df.shape)
df.to_csv('cleaned_data.csv', index=False)





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,
                              classification_report, roc_curve, roc_auc_score,
                              precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt

RANDOM_STATE = 42
df = pd.read_csv('cleaned_data.csv')

# ---------- Task 1: Labels ----------
y_reg = df['balance'].astype(float)
y_clf = (df['y'] == 'yes').astype(int)
drop_cols = ['balance', 'y']
if 'duration' in df.columns:
    drop_cols.append('duration')
X = df.drop(columns=drop_cols)

# ---------- Task 2: Encode categoricals ----------
ordinal_maps = {'education': {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}}
low_card_nominal = ['marital', 'default', 'housing', 'loan', 'contact', 'poutcome']
high_card_nominal = ['job', 'month']

X_enc = X.copy()
for col, mapping in ordinal_maps.items():
    X_enc[col] = X_enc[col].map(mapping)

X_enc = pd.get_dummies(X_enc, columns=low_card_nominal, drop_first=True)

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_arr = ohe.fit_transform(X_enc[high_card_nominal])
ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(high_card_nominal), index=X_enc.index)
X_enc = pd.concat([X_enc.drop(columns=high_card_nominal), ohe_df], axis=1)

# ---------- Task 3: Leak-free split + scaling ----------
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_enc, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

scaler = StandardScaler()
scaler.fit(X_train)  # fit ONLY on training data
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---------- Task 4: Regression ----------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)
mse = mean_squared_error(y_reg_test, y_pred_reg)
r2 = r2_score(y_reg_test, y_pred_reg)
print(f"LinearRegression -> MSE: {mse:,.2f}, R2: {r2:.4f}")

coefs = pd.Series(lin_reg.coef_, index=X_train_scaled.columns).sort_values(key=abs, ascending=False)
print(coefs.head(10))

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"Ridge(alpha=1.0) -> MSE: {mse_ridge:,.2f}, R2: {r2_ridge:.4f}")

# ---------- Task 5: Classification ----------
log_reg = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_clf_train)
y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

print(confusion_matrix(y_clf_test, y_pred_clf))
print(classification_report(y_clf_test, y_pred_clf, digits=3))

fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
auc = roc_auc_score(y_clf_test, y_proba_clf)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()
print("AUC:", auc)

# ---------- Task 6: Threshold sensitivity ----------
rows = []
for t in np.arange(0.30, 0.71, 0.05):
    preds_t = (y_proba_clf >= t).astype(int)
    rows.append([round(t,2),
                 precision_score(y_clf_test, preds_t, zero_division=0),
                 recall_score(y_clf_test, preds_t, zero_division=0),
                 f1_score(y_clf_test, preds_t, zero_division=0)])
threshold_df = pd.DataFrame(rows, columns=['Threshold','Precision','Recall','F1'])
print(threshold_df)

# ---------- Task 7: Regularization experiment ----------
log_reg_strong = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
log_reg_strong.fit(X_train_scaled, y_clf_train)
y_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]
y_pred_strong = log_reg_strong.predict(X_test_scaled)

log_reg_base = log_reg  # C=1.0 baseline already fit above
y_proba_base = y_proba_clf
y_pred_base = y_pred_clf

reg_table = pd.DataFrame({
    'Model': ['C=0.01 (strong L2)', 'C=1.0 (baseline)'],
    'Precision': [precision_score(y_clf_test, y_pred_strong, zero_division=0),
                  precision_score(y_clf_test, y_pred_base, zero_division=0)],
    'Recall': [recall_score(y_clf_test, y_pred_strong, zero_division=0),
               recall_score(y_clf_test, y_pred_base, zero_division=0)],
    'AUC': [roc_auc_score(y_clf_test, y_proba_strong),
            roc_auc_score(y_clf_test, y_proba_base)]
})
print(reg_table)

# ---------- Task 8: Bootstrap CI for AUC difference ----------
n_boot = 500
rng = np.random.default_rng(RANDOM_STATE)
y_clf_test_arr = y_clf_test.values
idx_all = np.arange(len(y_clf_test_arr))
diffs = []
for _ in range(n_boot):
    sample_idx = rng.choice(idx_all, size=len(idx_all), replace=True)
    y_true_s = y_clf_test_arr[sample_idx]
    if len(np.unique(y_true_s)) < 2:
        continue
    auc_base_s = roc_auc_score(y_true_s, y_proba_base[sample_idx])
    auc_strong_s = roc_auc_score(y_true_s, y_proba_strong[sample_idx])
    diffs.append(auc_base_s - auc_strong_s)

diffs = np.array(diffs)
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
print(f"Mean AUC diff: {diffs.mean():.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print("CI excludes 0:", not (ci_low <= 0 <= ci_high))
