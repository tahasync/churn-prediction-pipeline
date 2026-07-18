import numpy as np
import pandas as pd
import joblib
import os
import shap
from src.features import engineer_features


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")


class ChurnPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.pipeline = joblib.load(MODEL_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        self.classifier = self.pipeline.named_steps["classifier"]
        self.preprocessor = self.pipeline.named_steps["preprocessor"]

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        df = engineer_features(df)

        processed = self.preprocessor.transform(df)
        proba = self.pipeline.predict_proba(df)[0][1]
        pred = int(self.pipeline.predict(df)[0])

        top_features = self._compute_shap(processed)

        return {
            "churn_probability": round(float(proba), 4),
            "prediction": pred,
            "top_features": top_features,
        }

    def _compute_shap(self, processed):
        try:
            if hasattr(self.classifier, "feature_importances_"):
                explainer = shap.TreeExplainer(self.classifier)
            else:
                explainer = shap.LinearExplainer(self.classifier, processed)

            shap_values = explainer.shap_values(processed)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            sv = np.array(shap_values[0]).flatten()
            importance = np.abs(sv)
            top_idx = np.argsort(importance)[::-1][:3]

            result = []
            for idx in top_idx:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                result.append({"feature": name, "shap_value": round(float(sv[idx]), 4)})
            return result
        except Exception as e:
            return self._fallback_shap(processed, e)

    def _fallback_shap(self, processed, original_error):
        try:
            import shap
            explainer = shap.Explainer(self.classifier, processed)
            shap_values = explainer(processed)
            sv = shap_values.values[0]
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.array(sv).flatten()
            importance = np.abs(sv)
            top_idx = np.argsort(importance)[::-1][:3]
            result = []
            for idx in top_idx:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                result.append({"feature": name, "shap_value": round(float(sv[idx]), 4)})
            return result
        except Exception:
            feature_importance = np.abs(self.classifier.feature_importances_).flatten() if hasattr(self.classifier, "feature_importances_") else np.abs(self.classifier.coef_[0]).flatten()
            top_idx = np.argsort(feature_importance)[::-1][:3]
            result = []
            for idx in top_idx:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                result.append({"feature": name, "shap_value": round(float(feature_importance[idx]), 4)})
            return result


predictor = ChurnPredictor()
