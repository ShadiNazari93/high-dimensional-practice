import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from boruta import BorutaPy

# Optional: Bayesian optimization with scikit-optimize (skopt)
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import statsmodels.api as sm


# Define a class to encapsulate the full workflow

class MyocardialInfarctionPipeline:
    def __init__(self, ucid=579, target='FIBR_JELUD', random_state=42):
        self.ucid = ucid
        self.target = target
        self.random_state = random_state
        # Numerical columns known from dataset documentation
        self.numerical_col = ['AGE', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD',
                              'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE']
        self.categorical_cols = None
        # Outcome columns for imputation (all targets)
        self.outcome_cols = ['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC',
                             'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS']

        # Will be set later
        self.X = None
        self.y = None
        self.X_processed = None
        self.selected_features = None
        self.final_features = None
        self.models = {}

    def load_and_preprocess(self):
        """Load dataset, impute missing values, cap outliers, scale and encode features."""
        # Fetch dataset
        dataset = fetch_ucirepo(id=self.ucid)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        # Detect categorical columns
        self.categorical_cols = [col for col in X.columns if col not in self.numerical_col]

        # Impute missing values
        num_imputer = SimpleImputer(strategy='median')
        X.loc[:, self.numerical_col] = num_imputer.fit_transform(X[self.numerical_col])

        cat_imputer = SimpleImputer(strategy='most_frequent')
        X.loc[:, self.categorical_cols] = cat_imputer.fit_transform(X[self.categorical_cols])

        y_imputer = SimpleImputer(strategy='most_frequent')
        y_imputed = pd.DataFrame(y_imputer.fit_transform(y), columns=self.outcome_cols)

        # Outlier capping via IQR method for numerical columns
        def cap_outliers_iqr(df_subset):
            Q1 = df_subset.quantile(0.25)
            Q3 = df_subset.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return df_subset.clip(lower=lower, upper=upper, axis=1)

        X.loc[:, self.numerical_col] = cap_outliers_iqr(X[self.numerical_col])

        # Scale numerical features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(X[self.numerical_col])
        X_scaled = pd.DataFrame(scaled_features, columns=self.numerical_col)

        # Remove zero variance numerical features
        var_thresh = VarianceThreshold()
        X_num_var = var_thresh.fit_transform(X_scaled)
        selected_num_columns = X_scaled.columns[var_thresh.get_support()]

        # Encode categorical columns using OrdinalEncoder
        encoder = OrdinalEncoder()
        X_encoded_cat = pd.DataFrame(encoder.fit_transform(X[self.categorical_cols]), columns=self.categorical_cols)

        # Combine processed numerical and categorical features
        X_processed = pd.concat([X_scaled[selected_num_columns].reset_index(drop=True),
                                 X_encoded_cat.reset_index(drop=True)], axis=1)

        # Set class attributes
        self.X = X
        self.y = y_imputed
        self.X_processed = X_processed
        self.selected_num_columns = selected_num_columns
        self.encoder = encoder
        self.scaler = scaler

        print(f"Loaded and preprocessed dataset with shape: {self.X_processed.shape}")

    def feature_selection(self):
        """Run multiple feature selection methods and combine results."""
        target_y = self.y[self.target]

        # Variance Threshold to remove low-variance features (threshold example 0.01)
        var_thresh = VarianceThreshold(threshold=0.01)
        X_var_filt = var_thresh.fit_transform(self.X_processed)
        features_var_filt = self.X_processed.columns[var_thresh.get_support()]

        # Boruta selection
        rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5,
                                           random_state=self.random_state)
        boruta_selector = BorutaPy(estimator=rf_boruta, n_estimators='auto', random_state=self.random_state,
                                   max_iter=100)
        boruta_selector.fit(self.X_processed.values, target_y.values)
        boruta_features = self.X_processed.columns[boruta_selector.support_].to_list()

        # RFE with Logistic Regression
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state)
        rfe_selector = RFE(lr, n_features_to_select=10)
        rfe_selector.fit(self.X_processed, target_y)
        rfe_features = self.X_processed.columns[rfe_selector.support_].to_list()

        # LASSO with cross-validation
        X_scaled = self.scaler.transform(self.X_processed)  # scale full dataset
        lasso = LassoCV(cv=5, random_state=self.random_state)
        lasso.fit(X_scaled, target_y)
        lasso_coef = pd.Series(lasso.coef_, index=self.X_processed.columns)
        lasso_features = lasso_coef[lasso_coef != 0].index.to_list()

        # Combine all selected features (union)
        combined_features = list(set(boruta_features) | set(rfe_features) | set(lasso_features))
        self.selected_features = {
            'variance_threshold': features_var_filt.to_list(),
            'boruta': boruta_features,
            'rfe': rfe_features,
            'lasso': lasso_features,
            'combined': combined_features
        }

        print("Feature selection summary:")
        for method, feats in self.selected_features.items():
            print(f"{method}: {len(feats)} features")

    def train_and_evaluate_models(self):
        """Train models on selected features and evaluate with cross-validation."""
        target_y = self.y[self.target]
        X_data = self.X_processed[self.selected_features['combined']]

        # Train-test split stratified by target
        X_train, X_test, y_train, y_test = train_test_split(X_data, target_y,
                                                            test_size=0.3,
                                                            random_state=self.random_state,
                                                            stratify=target_y)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # Logistic Regression on combined features
        logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state)
        logreg.fit(X_train, y_train)
        self.models['logreg'] = logreg

        # Random Forest
        rf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_estimators=100)
        rf.fit(X_train, y_train)
        self.models['rf'] = rf

        # Cross-validation function
        def evaluate(model, X, y):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            print(f"{model.__class__.__name__} CV Accuracy: {acc.mean():.3f} (+/- {acc.std():.3f})")
            print(f"{model.__class__.__name__} CV AUC: {auc.mean():.3f} (+/- {auc.std():.3f})")

        print("Cross-validation performance on training set:")
        evaluate(logreg, X_train, y_train)
        evaluate(rf, X_train, y_train)

        # Test set performance
        y_pred = logreg.predict(X_test)
        y_prob = logreg.predict_proba(X_test)[:, 1]
        acc_test = accuracy_score(y_test, y_pred)
        auc_test = roc_auc_score(y_test, y_prob)
        print(f"Test Accuracy (LogReg): {acc_test:.3f}")
        print(f"Test AUC (LogReg): {auc_test:.3f}")

    # def shap_interpretation(self):
    #     """Compute and plot SHAP values for the logistic regression and random forest models."""
    #     # Use TreeExplainer for RF, KernelExplainer for LR
    #     explainer_rf = shap.TreeExplainer(self.models['rf'])
    #     shap_values_rf = explainer_rf.shap_values(self.X_test)
    #
    #     print("Plotting SHAP summary for Random Forest...")
    #     shap.summary_plot(shap_values_rf[1], self.X_test, plot_type="bar")
    #
    #     # For logistic regression, use KernelExplainer on a subset (due to computation)
    #     explainer_lr = shap.KernelExplainer(self.models['logreg'].predict_proba,
    #                                         shap.sample(self.X_train, 100, random_state=self.random_state))
    #     shap_values_lr = explainer_lr.shap_values(self.X_test.sample(50, random_state=self.random_state))
    #
    #     print("Plotting SHAP summary for Logistic Regression (50 samples)...")
    #     shap.summary_plot(shap_values_lr[1], self.X_test.sample(50, random_state=self.random_state), plot_type="bar")

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning with GridSearchCV and Bayesian Optimization on Logistic Regression."""
        X = self.X_processed[self.selected_features['combined']]
        y = self.y[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3,
                                                            random_state=self.random_state)

        # Grid search for LogisticRegression hyperparameters
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print("Grid Search best params:", grid_search.best_params_)

        # Bayesian Optimization for Random Forest
        param_space = {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 10)
        }
        rf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
        bayes_search = BayesSearchCV(rf, param_space, n_iter=20, cv=5, scoring='roc_auc', n_jobs=-1,
                                     random_state=self.random_state)
        bayes_search.fit(X_train, y_train)
        print("Bayesian Optimization best params:", bayes_search.best_params_)

        self.models['tuned_logreg'] = grid_search.best_estimator_
        self.models['tuned_rf'] = bayes_search.best_estimator_

    def nested_cross_validation(self):
        """Perform nested cross-validation to estimate generalization error."""
        import warnings
        warnings.filterwarnings("ignore")
        X = self.X_processed[self.selected_features['combined']]
        y = self.y[self.target]

        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        # Outer CV with inner grid search CV
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        scores = []
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Nested CV ROC AUC: {mean_score:.3f} (+/- {std_score:.3f})")

    def export_model(self, filename='final_model.pkl'):
        """Export the final selected logistic regression model and pipeline."""
        joblib.dump(self.models['logreg'], filename)
        print(f"Model exported to {filename}")

    def add_clinical_domain_features(self):
        """
        Placeholder function for incorporating clinical domain knowledge.
        Users can implement domain-driven feature construction such as ratios,
        binning age groups, interaction terms, etc.
        """
        # Example: Define age groups
        bins = [0, 40, 60, 80, 120]
        labels = ['<40', '40-59', '60-79', '80+']
        self.X_processed['AGE_GROUP'] = pd.cut(self.X['AGE'], bins=bins, labels=labels)

        # Re-encode newly created categorical feature
        encoded_age_group = self.encoder.fit_transform(self.X_processed[['AGE_GROUP']])
        self.X_processed['AGE_GROUP_ENC'] = encoded_age_group

        print("Clinical domain knowledge features added (age group)")

    def run_full_pipeline(self):
        """Execute all steps in order for a streamlined workflow."""
        self.load_and_preprocess()
        self.add_clinical_domain_features()
        self.feature_selection()
        self.train_and_evaluate_models()
        self.shap_interpretation()
        self.hyperparameter_tuning()
        self.nested_cross_validation()
        self.export_model()


# Usage Example
if __name__ == '__main__':
    pipeline = MyocardialInfarctionPipeline()
    pipeline.run_full_pipeline()
