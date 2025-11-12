from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from tabulate import tabulate

# 1. fetch dataset
myocardial_infarction_complications = fetch_ucirepo(id=579)

# 2. data (as pandas dataframes)
X = myocardial_infarction_complications.data.features.copy()
y = myocardial_infarction_complications.data.targets.copy()

# 3. variable information
print(myocardial_infarction_complications.variables)
print("Features shape:", X.shape)
print("Outcomes shape:", y.shape)

# 4. Define numerical and categorical columns
numerical_col = ['AGE','S_AD_KBRIG','D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD',
                 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD','L_BLOOD', 'ROE']

categorical_cols = [col for col in X.columns if col not in numerical_col]
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_col)

# 5. Impute missing values
num_imputer = SimpleImputer(strategy='median')
X.loc[:, numerical_col] = num_imputer.fit_transform(X[numerical_col])

cat_imputer = SimpleImputer(strategy='most_frequent')
X.loc[:, categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# 6. Impute missing values for y
outcome_cols = ['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC',
                'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS']
y_imputer = SimpleImputer(strategy='most_frequent')
y = pd.DataFrame(y_imputer.fit_transform(y), columns=outcome_cols)

# 7. Outlier capping function
def cap_outliers_iqr(df_subset):
    Q1 = df_subset.quantile(0.25)
    Q3 = df_subset.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df_subset.clip(lower=lower, upper=upper, axis=1)

# 8. Cap outliers in numerical columns
X.loc[:, numerical_col] = cap_outliers_iqr(X[numerical_col])

# 9. Scale numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[numerical_col])
X_scaled = pd.DataFrame(scaled_features, columns=numerical_col)

# 10. Remove zero variance features from numerical data
var_thresh = VarianceThreshold()
X_num_var = var_thresh.fit_transform(X_scaled)
selected_num_columns = X_scaled.columns[var_thresh.get_support()]

# Print dropped numerical columns (if any)
dropped_num_cols = set(numerical_col) - set(selected_num_columns)
if dropped_num_cols:
    print("Numerical columns dropped due to zero variance:", dropped_num_cols)
else:
    print("No numerical columns dropped due to zero variance.")

# Keep only selected numerical columns
X_selected_num = X_scaled[selected_num_columns]

# 11. Encode categorical columns with ordinal encoder
encoder = OrdinalEncoder()
X_encoded_cat = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=categorical_cols)

# 13. Combine processed numerical and categorical features into one DataFrame
X_processed = pd.concat([X_selected_num.reset_index(drop=True), X_encoded_cat.reset_index(drop=True)], axis=1)

# 14. Configure pandas display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 15. Show full processed features and outcome tables using tabulate
print("Processed Features (X_processed):")
print(tabulate(X_processed, headers='keys', tablefmt='psql'))
print("\nProcessed Outcomes (y):")
print(tabulate(y, headers='keys', tablefmt='psql'))
print("Processed Features shape:", X_processed.shape)
print("Processed Outcomes:", y.shape)

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assume X_processed and y['FIBR_JELUD'] are your processed features and target after your current preprocessing pipeline
# For demonstration, we extract target variable:
target = 'FIBR_JELUD'
y_selected = y[target]

# 1. Remove features with very low variance (already done partially, but full dataset check)
var_thresh = VarianceThreshold(threshold=0.01)  # Threshold can be adjusted
X_var_filtered = var_thresh.fit_transform(X_processed)
selected_features = X_processed.columns[var_thresh.get_support()]

print(f"Features retained after low variance filter: {list(selected_features)}")

# 2. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_processed[selected_features], y_selected, test_size=0.3, random_state=42, stratify=y_selected)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Check Multicollinearity using correlation matrix (Pearson)
plt.figure(figsize=(12,10))
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation matrix of features - Train set")
plt.show()

# Interpretation: High correlations (say > 0.8 or < -0.8) might indicate multicollinearity. You might drop or combine them.
# This is a visual step; changes depend on domain knowledge and correlation patterns.

# 4. Univariate Filter Methods
# Pearson Correlation (for continuous) vs target (binary classification)
# Use point biserial correlation approximation (since target is binary)
from scipy.stats import pointbiserialr
cont_features = numerical_col if set(numerical_col).issubset(set(X_train.columns)) else list(set(numerical_col).intersection(set(X_train.columns)))
print("Pearson correlation (point biserial) of continuous features with target:")
for feat in cont_features:
    corr, p = pointbiserialr(X_train[feat], y_train)
    print(f"{feat}: correlation={corr:.3f}, p-value={p:.4f}")

# Mutual Information for classification (any features)
mi_scores = mutual_info_classif(X_train, y_train, discrete_features=[col in categorical_cols for col in X_train.columns], random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
print("\nMutual Information Scores:")
print(mi_scores_series)

# Chi-square Test for categorical features
cat_features = [col for col in X_train.columns if col in categorical_cols]
print("\nChi-square test p-values for categorical features:")
from sklearn.feature_selection import chi2
X_cat = X_train[cat_features]
chi2_stats, p_values = chi2(X_cat, y_train)
for feat, p_val in zip(cat_features, p_values):
    print(f"{feat}: p-value = {p_val:.4e}")

# ANOVA F-test (for continuous features)
anova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(X_train[cont_features], y_train)
anova_scores = pd.Series(anova_selector.scores_, index=cont_features)
print("\nANOVA F-test scores for continuous features:")
print(anova_scores.sort_values(ascending=False))

# t-test / Wilcoxon is typically done for continuous features comparing target groups,
# here we show t-test example for each continuous feature
from scipy.stats import ttest_ind
print("\nt-test results for continuous features by target groups:")
for feat in cont_features:
    group0 = X_train[y_train == 0][feat]
    group1 = X_train[y_train == 1][feat]
    t_stat, p_val = ttest_ind(group0, group1, nan_policy='omit')
    print(f"{feat}: t-statistic={t_stat:.3f}, p-value={p_val:.4f}")

# Relief algorithm is not in sklearn; use `skrebate` or other packages (if allowed).
# If you want, I can show basic ReliefF using a package, but here is a note:
# Relief evaluates features based on how well their values distinguish between classes in neighboring instances.

# 5. Multivariate Filter: mRMR can be done using external libraries (not default sklearn)
# If permitted, a basic example using pymrmr:
# import pymrmr
# mrmr_features = pymrmr.mRMR(X_train.assign(target=y_train), 'MID', 10)
# print("Top features with mRMR:", mrmr_features)

# 6. Wrapper methods: Recursive Feature Elimination (RFE) with logistic regression
lr = LogisticRegression(max_iter=1000)
rfe_selector = RFE(estimator=lr, n_features_to_select=10, step=1)
rfe_selector.fit(X_train, y_train)
selected_rfe_features = X_train.columns[rfe_selector.support_]
print("\nSelected features by RFE with Logistic Regression:", list(selected_rfe_features))

# 7. Embedded Methods: LASSO Regularization with cross-validation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)
selected_lasso_features = lasso_coef[lasso_coef != 0].index.tolist()
print("\nSelected features by LASSO:", selected_lasso_features)

# 8. Tree-based feature importance (Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop features by Random Forest importance:")
print(importances.head(10))

# 9. Cross-validation for model performance with selected features

def evaluate_model(model, X_eval, y_eval):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X_eval, y_eval, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(model, X_eval, y_eval, cv=cv, scoring='roc_auc')
    print(f"Accuracy: {accuracy_scores.mean():.3f} (+/- {accuracy_scores.std():.3f})")
    print(f"AUC: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")
print("\nCross-validation performance with RFE-selected features:")
evaluate_model(LogisticRegression(max_iter=1000), X_train[selected_rfe_features], y_train)

print("\nCross-validation performance with LASSO-selected features:")
evaluate_model(LogisticRegression(max_iter=1000), X_train[selected_lasso_features], y_train)

print("\nCross-validation performance with Random Forest top 10 features:")
top_rf_features = importances.head(10).index.tolist()
evaluate_model(RandomForestClassifier(random_state=42), X_train[top_rf_features], y_train)

# 10. Final best feature selection: combining results
final_features = list(set(selected_rfe_features) | set(selected_lasso_features) | set(top_rf_features))
print("\nCombined final selected features:")
print(final_features)

# 11. Evaluate on test data with final features
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train[final_features], y_train)
y_pred = final_model.predict(X_test[final_features])
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, final_model.predict_proba(X_test[final_features])[:,1])
print(f"\nTest set accuracy with final features: {acc:.3f}")
print(f"Test set AUC with final features: {auc:.3f}")

# 12. Present final dataset after feature selection
X_final = X_processed[final_features]
print("Final dataset shape after feature selection:", X_final.shape)

# You can save or proceed with modeling with X_final and y_selected as needed
