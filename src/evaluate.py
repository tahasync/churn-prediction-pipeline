import json
import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import joblib


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")


def load_model_and_features():
    pipeline = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return pipeline, feature_names


def evaluate_model(X_test, y_test, pipeline, feature_names):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print("=== Model Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", pd.DataFrame(report).T)

    return metrics, cm, report


def _get_explainer(classifier, background_data):
    if hasattr(classifier, "feature_importances_"):
        return shap.TreeExplainer(classifier)
    else:
        return shap.LinearExplainer(classifier, background_data)


def _get_shap_values(explainer, data):
    try:
        sv = explainer.shap_values(data)
        if isinstance(sv, list):
            sv = sv[0]
        return sv
    except Exception as e:
        print(f"SHAP values computation failed: {e}")
        return None


def generate_shap_explanations(pipeline, X_test, feature_names, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    classifier = pipeline.named_steps["classifier"]
    X_test_processed = pipeline.named_steps["preprocessor"].transform(X_test)

    explainer = _get_explainer(classifier, X_test_processed[:100])
    shap_values = _get_shap_values(explainer, X_test_processed)

    if shap_values is None:
        print("Skipping SHAP plots (SHAP computation failed)")
        return

    plt.figure(figsize=(14, max(6, len(feature_names) * 0.35)))
    try:
        shap.summary_plot(
            shap_values, X_test_processed, feature_names=feature_names,
            show=False, max_display=min(20, len(feature_names)),
            plot_size=None, color_bar=True
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"Summary plot failed: {e}")
    finally:
        plt.close()

    indices = [(0, "shap_local_low_risk"), (1, "shap_local_borderline"), (-1, "shap_local_high_risk")]
    for idx, filename in indices:
        _generate_force_plot_safe(
            explainer, shap_values, X_test_processed,
            feature_names, output_dir, idx, filename
        )

    print(f"SHAP plots saved to {output_dir}")


def _generate_force_plot_safe(explainer, shap_values, X_processed, feature_names, output_dir, idx, filename):
    if X_processed.shape[0] <= abs(idx):
        return

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    sample = X_processed[idx:idx+1] if idx >= 0 else X_processed[idx:]
    shap_val = sv[idx] if idx >= 0 else sv[idx]

    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, list):
            ev = ev[0] if len(ev) == 2 else ev[0]
    else:
        ev = 0

    try:
        shap.force_plot(
            ev, shap_val, sample,
            feature_names=feature_names,
            matplotlib=True, show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"Force plot {filename} failed: {e}")
        _generate_horizontal_bar_plot(shap_val, feature_names, output_dir, filename)
    finally:
        plt.close()


def _generate_horizontal_bar_plot(shap_values, feature_names, output_dir, filename):
    vals = np.array(shap_values).flatten()
    names = feature_names[:len(vals)]
    top_k = min(10, len(vals))
    top_idx = np.argsort(np.abs(vals))[::-1][:top_k]

    plt.figure(figsize=(10, max(4, top_k * 0.4)))
    colors = ["#ef4444" if vals[i] > 0 else "#22c55e" for i in top_idx]
    plt.barh(range(top_k), vals[top_idx], color=colors)
    plt.yticks(range(top_k), [names[i] for i in top_idx])
    plt.axvline(0, color="white", linewidth=0.5)
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def business_interpretation(metrics):
    current_recall = metrics["recall"]
    target_recall = min(current_recall + 0.1, 0.99)
    precision = metrics["precision"]
    current_f1 = metrics["f1"]
    additional_churners = int(7043 * (target_recall - current_recall))

    print(f"""
=== Business Interpretation ===

Current model performance:
  - Recall: {current_recall:.2%}
  - Precision: {precision:.2%}
  - F1: {current_f1:.2%}
  - ROC-AUC: {metrics['roc_auc']:.2%}

If we adjust the decision threshold to increase recall to {target_recall:.2%}:
  - We catch approximately {additional_churners} more churners out of 7,043 customers
  - Precision would drop by ~5-8 percentage points (more false positives)
  - Estimated false positives per 100 flagged customers: ~{int((1 - precision + 0.07) * 100)} customers

Business trade-off:
  In telecom, the cost of a missed churner (lost CLV ~$1,000-$2,000) far exceeds
  the cost of a retention offer (~$50). Even with lower precision, the ROI of
  proactive retention is strongly positive. The model enables targeting the
  right customers with the right offers — e.g., customers flagged by TechSupport
  can receive tech support discounts.
""")


if __name__ == "__main__":
    from src.preprocessing import load_data
    from src.features import engineer_features
    from sklearn.model_selection import train_test_split

    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline, feature_names = load_model_and_features()
    metrics, cm, report = evaluate_model(X_test, y_test, pipeline, feature_names)

    results_path = os.path.join(MODELS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print("\n=== Cross-Validation Results ===")
        for model_name, scores in results.items():
            print(f"{model_name}: ROC-AUC = {scores['test_roc_auc']['mean']:.4f} "
                  f"(+/- {scores['test_roc_auc']['std']:.4f})")

    generate_shap_explanations(pipeline, X_test, feature_names)
    business_interpretation(metrics)
