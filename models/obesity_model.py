from __future__ import annotations

import pandas as pd

import os
import joblib
from sklearn.pipeline import Pipeline

from schemas.obesity_schema import ObesityPredictionInput, ModelPrediction, CombinedPredictionOutput, \
    ObesityPredictionOutput

memory = joblib.Memory(location="cache.joblib")

@memory.cache(ignore=["model"])
def predict(model: Pipeline, input_data: pd.DataFrame, threshold: float = 0.5) -> tuple[str, float]:
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data

    probability = model.predict_proba(df)[0][1]
    prediction = "obese" if probability >= threshold else "non-obese"

    return prediction, probability


class ObesityModel:
    logistic_model: Pipeline | None = None
    cart_model: Pipeline | None = None
    naive_bayes_model: Pipeline | None = None
    feature_order: list[str] | None = None
    targets: list[str] | None = None

    def load_model(self) -> None:
        """Loads the obesity prediction models (Logistic Regression, CART, Naive Bayes)"""
        base_dir = os.path.dirname(__file__)

        logistic_file = os.path.join(base_dir, "logistic_model.joblib")
        self.logistic_model, feature_order = joblib.load(logistic_file)

        cart_file = os.path.join(base_dir, "cart_model.joblib")
        self.cart_model, _ = joblib.load(cart_file)

        naive_bayes_file = os.path.join(base_dir, "naive_bayes_model.joblib")
        self.naive_bayes_model, _ = joblib.load(naive_bayes_file)

        self.feature_order = feature_order
        self.targets = ["non-obese", "obese"]



    def predict_obesity(self, prediction_input: ObesityPredictionInput) -> ObesityPredictionOutput:
        """Runs an obesity prediction using the logistic regression model"""
        if not self.logistic_model or not self.feature_order:
            raise RuntimeError("Model is not loaded")

        input_dict = prediction_input.model_dump()
        input_df = pd.DataFrame([input_dict], columns=self.feature_order)

        prediction, probability = predict(self.logistic_model, input_df)

        return ObesityPredictionOutput(
            obesity_status=prediction,
            probability=round(probability, 4)
        )


    def predict(self, model: Pipeline, input_df: pd.DataFrame) -> ModelPrediction:
        """Runs a prediction for a single model"""
        if not model or not self.feature_order:
            raise RuntimeError("Model is not loaded")

        prediction, probability = predict(model, input_df)
        return ModelPrediction(obesity_status=prediction, probability=round(probability, 4))


    def predict_all(self, prediction_input: ObesityPredictionInput) -> CombinedPredictionOutput:
        """Runs predictions for all three models"""
        if not self.logistic_model or not self.cart_model or not self.naive_bayes_model or not self.feature_order:
            raise RuntimeError("Models are not loaded")

        input_dict = prediction_input.model_dump()
        input_df = pd.DataFrame([input_dict], columns=self.feature_order)

        logistic_prediction = self.predict(self.logistic_model, input_df)
        cart_prediction = self.predict(self.cart_model, input_df)
        naive_bayes_prediction = self.predict(self.naive_bayes_model, input_df)

        return CombinedPredictionOutput(
            logistic=logistic_prediction,
            cart=cart_prediction,
            naive_bayes=naive_bayes_prediction
        )


obesity_model = ObesityModel()
