from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import statsmodels.api as sm
from tabulate import tabulate
from scipy.stats import pointbiserialr, ttest_ind


# 1. Load data
data_ucirepo = fetch_ucirepo(id=579)
X = data_ucirepo.data.features.copy()
y = data_ucirepo.data.targets.copy()
print("Features shape:", X.shape)
print("Outcomes shape:", y.shape)
# Define columns
numerical_col = ['AGE', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD',
                 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE']
categorical_cols = [c for c in X.columns if c not in numerical_col]
outcome_cols = ['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC',
                'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS']
# 2. Impute missing values
X[numerical_col] = SimpleImputer(strategy='median').fit_transform(X[numerical_col])
X[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_cols])
y = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(y), columns=outcome_cols)
# 3. Outlier capping (IQR method)
def cap_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df.clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR, axis=1)
X[numerical_col] = cap_outliers_iqr(X[numerical_col])
# 4. Scale numerical features
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_col]), columns=numerical_col)
# 5. Encode categorical variables
encoder = OrdinalEncoder()
X_cat_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=categorical_cols)
# 6. Remove low variance features
var_thresh = VarianceThreshold(threshold=0.01)
X_num_scaled= pd.DataFrame(var_thresh.fit_transform(X_num_scaled), columns=X_num_scaled.columns[var_thresh.get_support()])
# Print dropped numerical columns (if any)
dropped_num_cols = set(numerical_col) - set(X_num_scaled)
if dropped_num_cols:
    print("Numerical columns dropped due to zero variance:", dropped_num_cols)
else:
    print("No numerical columns dropped due to zero variance.")

# 7. Combine numerical and categorical
X_processed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

# Assume X_processed and y['FIBR_JELUD'] are your processed features and target after your current preprocessing pipeline
# 8. Select target for modeling, for demonstration
target = 'FIBR_JELUD'
y_selected = y[target]
# 9. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_selected, stratify=y_selected, test_size=0.3, random_state=42)

# 10. Check Multicollinearity using correlation matrix (Pearson)
plt.figure(figsize=(12,10))
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title("Correlation matrix of features - Train set")
plt.show()
# Interpretation: High correlations (say > 0.8 or < -0.8) might indicate multicollinearity. You might drop or combine them.
# This is a visual step; changes depend on domain knowledge and correlation patterns.

# 11. Univariate Filter Methods
# Pearson Correlation (for continuous) vs target (binary classification)
# Use point biserial correlation approximation (since target is binary)
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
print("\nt-test results for continuous features by target groups:")
for feat in cont_features:
    group0 = X_train[y_train == 0][feat]
    group1 = X_train[y_train == 1][feat]
    t_stat, p_val = ttest_ind(group0, group1, nan_policy='omit')
    print(f"{feat}: t-statistic={t_stat:.3f}, p-value={p_val:.4f}")

# 12. Wrapper methods: Recursive Feature Elimination (RFE) with logistic regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rfe_selector = RFE(estimator=lr, n_features_to_select=10, step=1)
rfe_selector.fit(X_train, y_train)
selected_rfe_features = X_train.columns[rfe_selector.support_]
print("\nSelected features by RFE with Logistic Regression:", list(selected_rfe_features))

# 13. Embedded Methods: LASSO Regularization with cross-validation
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)
selected_lasso_features = lasso_coef[lasso_coef != 0].index.tolist()
print("\nSelected features by LASSO:", selected_lasso_features)

# 14. Tree-based feature importance (Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop features by Random Forest importance:")
print(importances.head(10))

# 15. Boruta feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter=50)
boruta_selector.fit(X_train.values, y_train.values.ravel())
boruta_features = X_train.columns[boruta_selector.support_].tolist()
print("Boruta selected features:", boruta_features)

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Define top filter features based on mutual_info or ANOVA scores (example: top 10 MI features)
top_mi_features = mi_scores_series.head(10).index.to_list()
top_anova_features = anova_scores.sort_values(ascending=False).head(10).index.to_list()

# Prepare all feature sets
feature_sets = {
    'RFE': set(selected_rfe_features),
    'LASSO': set(selected_lasso_features),
    'Boruta': set(boruta_features),
    'MutualInfo Filter': set(top_mi_features),
    'ANOVA Filter': set(top_anova_features)
}

# Venn plot for three main sets (RFE, LASSO, Boruta)
plt.figure(figsize=(8,8))
venn3([feature_sets['RFE'], feature_sets['LASSO'], feature_sets['Boruta']],
      set_labels=('RFE', 'LASSO', 'Boruta'))
plt.title('Venn Diagram of Feature Selection Methods')
plt.show()

# Random Forest feature importance (top 15)
plt.figure(figsize=(12,6))
sns.barplot(x=importances.head(15).values, y=importances.head(15).index, palette='viridis')
plt.title('Random Forest Feature Importance (Top 15)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# LASSO Coefficients plot (nonzero)
plt.figure(figsize=(12,6))
lasso_nonzero = lasso_coef[lasso_coef != 0].sort_values()
lasso_nonzero.plot(kind='barh', color='teal')
plt.title('LASSO Non-zero Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# Fit logistic regression on RFE features for odds ratio plot
X_train_sm = sm.add_constant(X_train[selected_rfe_features])
logit_model = sm.Logit(y_train, X_train_sm).fit(disp=False)
odds_ratios = pd.DataFrame({
    'OR': np.exp(logit_model.params),
    'Lower CI': np.exp(logit_model.conf_int()[0]),
    'Upper CI': np.exp(logit_model.conf_int()[1])
}).loc[selected_rfe_features]

plt.figure(figsize=(10,6))
plt.errorbar(odds_ratios['OR'], odds_ratios.index,
             xerr=[odds_ratios['OR'] - odds_ratios['Lower CI'], odds_ratios['Upper CI'] - odds_ratios['OR']],
             fmt='o', color='black')
plt.axvline(1, color='red', linestyle='--')
plt.title('Forest plot: Odds Ratios and 95% CI for RFE features')
plt.xlabel('Odds Ratio')
plt.ylabel('Features')
plt.show()

# Nomogram conceptual plot (scaled logistic regression coefficients)
coefs = logit_model.params[selected_rfe_features]
coefs_scaled = (coefs - coefs.min()) / (coefs.max() - coefs.min())
plt.figure(figsize=(10,6))
plt.barh(coefs.index, coefs_scaled, color='skyblue')
plt.title('Nomogram Conceptual Plot (Scaled Coefficients)')
plt.xlabel('Scaled Coefficient (0-1)')
plt.ylabel('Features')
plt.show()

# Decision Curve Analysis (DCA)
thresholds = np.linspace(0, 1, 100)
p_outcome = y_train.mean()
prob_pred = logit_model.predict(X_train_sm)

net_benefit_model = []
net_benefit_all = []
net_benefit_none = []

for pt in thresholds:
    sens = ((prob_pred >= pt) & (y_train == 1)).sum() / (y_train == 1).sum()
    spec = ((prob_pred < pt) & (y_train == 0)).sum() / (y_train == 0).sum()
    nb = sens * p_outcome - (1 - spec) * (1 - p_outcome) * (pt / (1 - pt))
    net_benefit_model.append(nb)
    net_benefit_all.append(p_outcome - (1 - pt) * (1 - p_outcome))
    net_benefit_none.append(0)

plt.figure(figsize=(10,6))
plt.plot(thresholds, net_benefit_model, label='Model')
plt.plot(thresholds, net_benefit_all, label='Treat All', linestyle='--')
plt.plot(thresholds, net_benefit_none, label='Treat None', linestyle='--')
plt.title('Decision Curve Analysis')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.legend()
plt.show()

# Calibration Curve on test set
prob_pred_test = logit_model.predict(sm.add_constant(X_test[selected_rfe_features]))
prob_true, prob_pred_cal = calibration_curve(y_test, prob_pred_test, n_bins=10)

plt.figure(figsize=(8,8))
plt.plot(prob_pred_cal, prob_true, marker='o', label='Calibration Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()

# Model evaluation and best feature selection
results = []
for method, feats in feature_sets.items():
    feats = list(feats)
    # Skip if empty
    if len(feats) == 0:
        continue
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train[feats], y_train)
    pred_train = model.predict(X_train[feats])
    pred_prob = model.predict_proba(X_train[feats])[:,1]
    cv_scores = cross_val_score(model, X_train[feats], y_train, cv=5, scoring='accuracy')
    results.append({
        'Method': method,
        'Num Features': len(feats),
        'Train Accuracy': accuracy_score(y_train, pred_train),
        'Train AUC': roc_auc_score(y_train, pred_prob),
        'CV Accuracy Mean': cv_scores.mean()
    })

results_df = pd.DataFrame(results)
print("\nModel evaluation results for each feature selection method:")
print(results_df)

# Best method by highest cross-validation accuracy
best_method_row = results_df.loc[results_df['CV Accuracy Mean'].idxmax()]
print(f"\nBest feature selection method: {best_method_row['Method']} with CV accuracy = {best_method_row['CV Accuracy Mean']:.3f}")

best_features = list(feature_sets[best_method_row['Method']])
print("Best features selected:", best_features)

# Reduce selected dataset
X_train_selected = X_train[best_features]
X_test_selected = X_test[best_features]

# 14. Fit final model with statsmodels Logit
X_train_final = sm.add_constant(X_train[best_features])
try:
    logit_model = sm.Logit(y_train, X_train_final)
    result = logit_model.fit(disp=False)
except Exception as e:
    print("Statsmodels Logit fit failed:", e)
    result = None

if result:
    # Odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    conf_exp = np.exp(conf)
    print("\nOdds Ratios with 95% Confidence Intervals:\n", conf_exp)

    plt.errorbar(conf_exp['OR'], range(len(conf_exp)), xerr=[conf_exp['OR']-conf_exp['2.5%'], conf_exp['97.5%']-conf_exp['OR']], fmt='o')
    plt.yticks(range(len(conf_exp)), conf_exp.index)
    plt.axvline(1, color='grey', linestyle='--')
    plt.xlabel('Odds Ratio')
    plt.title('Odds Ratios with 95% Confidence Intervals')
    plt.tight_layout()
    plt.show()

# 15. Model evaluation on test set using sklearn LogisticRegression
final_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
final_model.fit(X_train[best_features], y_train)
y_pred = final_model.predict(X_test[best_features])
y_prob = final_model.predict_proba(X_test[best_features])[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test AUC: {roc_auc_score(y_test, y_prob):.3f}")

# 16. Calibration curve plot
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.show()

# 17. Decision curve analysis
thresholds = np.linspace(0, 1, 100)
net_benefit = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    nb = (tp/len(y_test)) - (fp/len(y_test)) * (t / (1 - t))
    net_benefit.append(nb)
plt.plot(thresholds, net_benefit, label='Model')
plt.plot(thresholds, np.zeros_like(thresholds), linestyle='--', label='No Intervention')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.tight_layout()
plt.show()

# 18. Summary print using tabulate
print("Final selected features and their data preview:")
print(tabulate(X_train[best_features].head(), headers='keys', tablefmt='psql'))
print("Processed outcome preview:")
print(tabulate(y_selected.head().to_frame(), headers='keys', tablefmt='psql'))
