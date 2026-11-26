import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from config import RANDOM_STATE, XGB_PARAMS


def train_and_evaluate_xgboost(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = XGBClassifier(**XGB_PARAMS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))

    print("\n[classifier] XGBoost Evaluation")
    print(f"  5-Fold CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Held-Out Macro F1:  {test_f1:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, float(cv_scores.mean()), test_f1, importances
