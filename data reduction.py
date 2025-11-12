import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# --- Load Data ---
expr_df = pd.read_csv(EXPR_PATH, index_col=0)
labels_df = pd.read_csv(LABELS_PATH)

# Map string labels ("ALL", "AML") to binary for classification
labels_dict = {"ALL": 0, "AML": 1}
labels_df["label"] = labels_df["diagnosis"].map(labels_dict)  # Adjust column name if necessary

# Match sample order
labels_df = labels_df.set_index("id").reindex(expr_df.index)
y = labels_df["label"].values
X = expr_df.values

# Impute missing values (if any)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_imputed)

# Remove zero-variance features
var_thresh = VarianceThreshold()
X_var = var_thresh.fit_transform(X_scaled)
selected_columns = expr_df.columns[var_thresh.get_support()]

# Limit to top 500 by variance
if X_var.shape[1] > 500:
    variances = np.var(X_var, axis=0)
    top_idx = np.argsort(variances)[::-1][:500]
    X_final = X_var[:, top_idx]
    selected_final = selected_columns[top_idx]
else:
    X_final = X_var
    selected_final = selected_columns

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=1000, solver="liblinear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test accuracy with top {X_final.shape[1]} features: {acc:.3f}")

# Save selected features for reference
pd.Series(selected_final).to_csv("selected_features.csv", index=False)
