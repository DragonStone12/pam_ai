from schemas.obesity_schema import ModelPrediction, ObesityPredictionOutput, CombinedPredictionOutput

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
    return ObesityPredictionOutput(
        obesity_status='obese',
        probability=1
    )


def mock_predictions() -> CombinedPredictionOutput:
    return CombinedPredictionOutput(
        logistic=logistic,
        cart=cart,
        naive_bayes=naive_bayes
    )
