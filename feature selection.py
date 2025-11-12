import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from boruta import BorutaPy
from sklearn.calibration import calibration_curve
from matplotlib_venn import venn2
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


target = 'FIBR_JELUD'
y_selected = y[target]

# 2. Recursive Feature Elimination (RFE) with Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rfe_selector = RFE(lr, n_features_to_select=10)
rfe_selector.fit(X_processed, y_selected)
rfe_features = X_processed.columns[rfe_selector.support_].to_list()
print("RFE selected features:", rfe_features)

# 3. Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_processed, y_selected)
rf_importances = pd.Series(rf.feature_importances_, index=X_processed.columns).sort_values(ascending=False)
print("Random Forest top features:\n", rf_importances.head(10))

plt.figure(figsize=(8,5))
sns.barplot(x=rf_importances.head(10), y=rf_importances.head(10).index)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# 4. LASSO feature selection
lasso = LassoCV(cv=5, random_state=42)
X_scaled = StandardScaler().fit_transform(X_processed)
lasso.fit(X_scaled, y_selected)
lasso_coef = pd.Series(lasso.coef_, index=X_processed.columns)
lasso_features = lasso_coef[lasso_coef != 0].index.to_list()
print("LASSO selected features:", lasso_features)

plt.figure(figsize=(8,5))
lasso_coef.sort_values().plot(kind='barh')
plt.title("LASSO Coefficients")
plt.tight_layout()
plt.show()

# 5. Venn Diagram of overlapping features selected by Boruta, RFE, and LASSO
plt.figure(figsize=(6,6))
venn2([set(rfe_features), set(lasso_features)],
      set_labels=('RFE', 'LASSO'))
plt.title("Overlap of Selected Features by RFE and LASSO")
plt.show()

# 6. Logistic Regression Odds Ratios and Confidence Intervals
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_selected, stratify=y_selected, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)

import statsmodels.api as sm
X_train_const = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Odds ratios and 95% confidence intervals
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
conf = np.exp(conf)
print(conf)

# Forest plot for ORs
or_vals = conf['OR']
lower = conf['2.5%']
upper = conf['97.5%']

plt.errorbar(or_vals, range(len(or_vals)), xerr=[or_vals - lower, upper - or_vals], fmt='o')
plt.yticks(range(len(or_vals)), conf.index)
plt.axvline(x=1, color='grey', linestyle='--')
plt.xlabel('Odds Ratio')
plt.title('Odds Ratios with 95% Confidence Intervals')
plt.tight_layout()
plt.show()

# 7. Nomogram Conceptual Plot (simplified, points proportional to coefficients)
features = conf.index[1:]  # exclude intercept
coeffs = conf.loc[features, 'OR']
points = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min()) * 100

plt.figure(figsize=(8,6))
plt.barh(features, points)
plt.xlabel("Nomogram Points (relative)")
plt.title("Simplified Nomogram Feature Points")
plt.tight_layout()
plt.show()

# 8. Decision Curve Analysis (simplified net benefit curve)
from sklearn.metrics import precision_recall_curve

prob_pos = log_reg.predict_proba(X_test)[:,1]
thresholds = np.linspace(0, 1, 100)
net_benefit = []
for t in thresholds:
    preds = (prob_pos >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    nb = (tp / len(y_test)) - (fp / len(y_test)) * (t / (1-t))
    net_benefit.append(nb)

plt.plot(thresholds, net_benefit, label='Nomogram (LogReg)')
plt.plot(thresholds, np.zeros_like(thresholds), label='None')
plt.plot(thresholds, [max(y_test.mean() - t/(1-t)*(1 - y_test.mean()),0) for t in thresholds], label='All')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, prob_pos, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Nomogram')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.show()

