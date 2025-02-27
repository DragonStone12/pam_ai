from models.obesity_model import weights, THRESHOLD
from schemas.obesity_schema import ModelPrediction, ObesityPredictionOutput

logistic = ModelPrediction(
    obesity_status='obese',
    probability=0.83
)

cart = ModelPrediction(
    obesity_status='obese',
    probability=0.71
)

naive_bayes = ModelPrediction(
    obesity_status='obese',
    probability=0.68
)


def mock_prediction() -> ObesityPredictionOutput:
    weighted_probability = (
        weights['logistic'] * logistic.probability +
        weights['cart'] * cart.probability +
        weights['naive_bayes'] * naive_bayes.probability
    )

    prediction = "obese" if weighted_probability >= THRESHOLD else "non-obese"

    return ObesityPredictionOutput(
        logistic=logistic,
        cart=cart,
        naive_bayes=naive_bayes,
        prediction=prediction,
        probability=round(weighted_probability, 4)
    )
