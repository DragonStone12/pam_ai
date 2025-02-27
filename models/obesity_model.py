from __future__ import annotations

import pandas as pd

import os
import joblib
from sklearn.pipeline import Pipeline

from schemas.obesity_schema import ObesityPredictionInput, ModelPrediction, ObesityPredictionOutput

memory = joblib.Memory(location="cache.joblib")

weights = {
    'logistic': 0.5,
    'cart': 0.3,
    'naive_bayes': 0.2
}

THRESHOLD = 0.65

@memory.cache(ignore=["model"])
def predict(model: Pipeline, input_data: pd.DataFrame) -> tuple[str, float]:
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data

    probability = model.predict_proba(df)[0][1]
    prediction = "obese" if probability >= THRESHOLD else "non-obese"

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
        if not self.logistic_model or not self.cart_model or not self.naive_bayes_model or not self.feature_order:
            raise RuntimeError("Models are not loaded")

        input_dict = prediction_input.model_dump()
        input_df = pd.DataFrame([input_dict], columns=self.feature_order)

        logistic_prediction, logistic_probability = predict(self.logistic_model, input_df)
        cart_prediction, cart_probability = predict(self.cart_model, input_df)
        naive_bayes_prediction, naive_bayes_probability = predict(self.naive_bayes_model, input_df)

        logistic_result = ModelPrediction(
            obesity_status=logistic_prediction,
            probability=round(logistic_probability, 4)
        )

        cart_result = ModelPrediction(
            obesity_status=cart_prediction,
            probability=round(cart_probability, 4)
        )

        naive_bayes_result = ModelPrediction(
            obesity_status=naive_bayes_prediction,
            probability=round(naive_bayes_probability, 4)
        )

        weighted_probability = (
            weights['logistic'] * logistic_probability +
            weights['cart'] * cart_probability +
            weights['naive_bayes'] * naive_bayes_probability
        )

        prediction = "obese" if weighted_probability >= THRESHOLD else "non-obese"

        return ObesityPredictionOutput(
            logistic=logistic_result,
            cart=cart_result,
            naive_bayes=naive_bayes_result,
            prediction=prediction,
            probability=round(weighted_probability, 4)
        )

obesity_model = ObesityModel()
