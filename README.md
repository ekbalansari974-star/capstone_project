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














import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

RANDOM_STATE = 42

# ============================================================
# Rebuild Part 2 state (labels, encoding, split, scaling)
# ============================================================
df = pd.read_csv('cleaned_data.csv')

y_reg = df['balance'].astype(float)
y_clf = (df['y'] == 'yes').astype(int)
drop_cols = ['balance', 'y']
if 'duration' in df.columns:
    drop_cols.append('duration')  # leakage guard
X = df.drop(columns=drop_cols)

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

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_enc, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print("Part 2 state rebuilt. Train/test shapes:", X_train_scaled.shape, X_test_scaled.shape)


# ============================================================
# Task 1 — Decision Tree baseline (no constraints)
# ============================================================
dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_base.fit(X_train_scaled, y_clf_train)

train_acc_base = accuracy_score(y_clf_train, dt_base.predict(X_train_scaled))
test_acc_base = accuracy_score(y_clf_test, dt_base.predict(X_test_scaled))
print(f"\n[Task 1] Baseline DT -> train acc: {train_acc_base:.4f}, test acc: {test_acc_base:.4f}")
# README: near-1.0 train acc with a visibly lower test acc = overfitting signature —
# the unconstrained tree grows leaves until pure, memorizing training noise.


# ============================================================
# Task 2 — Controlled Decision Tree
# ============================================================
dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=RANDOM_STATE)
dt_ctrl.fit(X_train_scaled, y_clf_train)

train_acc_ctrl = accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled))
print(f"[Task 2] Controlled DT -> train acc: {train_acc_ctrl:.4f}, test acc: {test_acc_ctrl:.4f}")
# README: max_depth caps how many splits deep the tree can go (limits complexity);
# min_samples_split blocks splitting a node with fewer than 10 samples
# (prevents splits on tiny, unreliable subsets). Both shrink the train/test gap.


# ============================================================
# Task 3 — Gini vs Entropy comparison
# ============================================================
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=RANDOM_STATE)
dt_gini.fit(X_train_scaled, y_clf_train)
acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=RANDOM_STATE)
dt_entropy.fit(X_train_scaled, y_clf_train)
acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print(f"[Task 3] Gini test acc: {acc_gini:.4f} | Entropy test acc: {acc_entropy:.4f}")
# README formulas: Gini = 1 - sum(p_i^2)   |   Entropy = -sum(p_i * log2(p_i))
# Both measure node impurity; 0 means a pure node (all one class).


# ============================================================
# Task 4 — Random Forest
# ============================================================
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"[Task 4] RF -> train acc: {rf_train_acc:.4f}, test acc: {rf_test_acc:.4f}, AUC: {rf_auc:.4f}")

importances = pd.Series(rf.feature_importances_, index=X_train_scaled.columns).sort_values(ascending=False)
print("Top 5 features:\n", importances.head(5))
# README: RF importance = average impurity decrease a feature contributes across
# all splits/trees. Differs from linear coefficients: always non-negative, captures
# non-linear/interaction effects, and dilutes across correlated features.


# ============================================================
# Task 4a — Gradient Boosting
# ============================================================
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
gb.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"[Task 4a] GB -> train acc: {gb_train_acc:.4f}, test acc: {gb_test_acc:.4f}, AUC: {gb_auc:.4f}")


# ============================================================
# Task 4b — Feature ablation study
# ============================================================
lowest5 = importances.tail(5).index.tolist()
print("[Task 4b] 5 lowest-importance features:", lowest5)

X_train_reduced = X_train_scaled.drop(columns=lowest5)
X_test_reduced = X_test_scaled.drop(columns=lowest5)

rf_reduced = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
rf_reduced.fit(X_train_reduced, y_clf_train)

auc_full = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
auc_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
print(f"[Task 4b] Full-feature AUC: {auc_full:.4f} | Reduced-feature AUC: {auc_reduced:.4f}")
# README: if auc_reduced ~= auc_full, those features were genuinely uninformative;
# if it drops noticeably, they were contributing real (if individually small) signal.


# ============================================================
# Task 5 — Cross-validated comparison
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree (max_depth=5)': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=skf, scoring='roc_auc')
    cv_results[name] = (scores.mean(), scores.std())
    print(f"[Task 5] {name}: mean AUC={scores.mean():.4f}, std={scores.std():.4f}")
# README: 5-fold CV averages AUC over 5 different splits, giving a lower-variance
# estimate of generalization than a single train/test split; std shows how
# sensitive that estimate is to which rows land in each fold.


# ============================================================
# Task 6 — GridSearchCV hyperparameter tuning
# ============================================================
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5],
}

pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=RANDOM_STATE)
)

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_clf_train)  # unscaled X_train -- pipeline handles imputing+scaling

print(f"[Task 6] Best params: {grid_search.best_params_}")
print(f"[Task 6] Best CV AUC: {grid_search.best_score_:.4f}")

total_configs = 1
for v in param_grid.values():
    total_configs *= len(v)
print(f"[Task 6] {total_configs} configs x 5 folds = {total_configs * 5} total fits")
# README: exhaustive grid search checks every combination (18 configs x 5 folds = 90
# fits here), guaranteeing the reported best is optimal within this grid, at a
# real compute cost — the trade-off vs. a lighter randomized search.

best_pipeline = grid_search.best_estimator_


# ============================================================
# Task 7 — Manual learning curve (using the tuned pipeline)
# ============================================================
learning_rows = []
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    n_rows = int(frac * len(X_train))
    X_sub = X_train.iloc[:n_rows]
    y_sub = y_clf_train.iloc[:n_rows]

    best_pipeline.fit(X_sub, y_sub)
    train_auc = roc_auc_score(y_sub, best_pipeline.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
    learning_rows.append([frac, n_rows, train_auc, test_auc])

learning_df = pd.DataFrame(learning_rows, columns=['Training Fraction', 'N Rows', 'Training AUC', 'Test AUC'])
print("[Task 7] Learning curve:\n", learning_df)
# README: (i) falling training AUC as data grows is expected — harder to fit every
# point with more data; (ii) rising test AUC with more data = more data would help;
# (iii) if test AUC still climbing at 100% -> data-limited; if flattened earlier
# -> capacity-limited, more data won't help further.

best_pipeline.fit(X_train, y_clf_train)  # refit on full training data before saving


# ============================================================
# Task 8 — Serialize the best model
# ============================================================
joblib.dump(best_pipeline, 'best_model.pkl')
print("[Task 8] Saved best_model.pkl")

# reload-and-predict verification block (README requires this to run without errors)
loaded_model = joblib.load('best_model.pkl')
sample_preds = loaded_model.predict(X_test.iloc[:5])
print("[Task 8] Reload + predict check:", sample_preds)


# ============================================================
# Task 9 — Summary comparison table (paste into README as Markdown)
# ============================================================
summary_rows = []
for name, (mean_auc, std_auc) in cv_results.items():
    summary_rows.append([name, mean_auc, std_auc])
summary_rows.append(['Tuned Random Forest (GridSearchCV)', grid_search.best_score_, np.nan])

summary_df = pd.DataFrame(summary_rows, columns=['Model', '5-Fold CV Mean AUC', '5-Fold CV Std AUC'])
print("\n[Task 9] Summary table:\n", summary_df.to_markdown(index=False))
# README: add a Test-Set AUC column (from each model's held-out X_test_scaled/X_test
# performance), then state your recommended model and justify in 3-5 sentences --
# usually the model with the best mean CV AUC *and* an acceptably low std.

print("\nALL PART 3 TASKS COMPLETE")
